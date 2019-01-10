require('torch')
require('nn')
require('optim')
require('paths')
require('nngraph')
require('image')
require('utilities')
require('DatasetLmdb')   -- data load
require('MainModel_recog')  -- main model
require('optim_updates')
require('hdf5')

local opts = require 'train_opts'
local cjson = require 'cjson'
-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  -- cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
  gpuidx = cutorch.getDevice()
  print(gpuidx)
end

-- loading training data Grop 1
ij=1
-- set the training data path
opt.trainSetPath='DataDB/train_' .. tostring(ij) .. '/data.mdb'
print(opt.trainSetPath)

local trainSet = DatasetLmdb(opt.trainSetPath, opt.batch_size)
local valSet = DatasetLmdb(opt.valSetPath, opt.batch_size)
local vocab_file = path.join(opt.token_dir, 'vocab.t7')
local token_file = path.join(opt.token_dir, 'token.t7')
opt.idx_to_token = torch.load(token_file)
opt.token_to_idx = torch.load(vocab_file)
opt.beam_size = nil

-- count vocab
local vocab_size = 0
for _ in pairs(opt.idx_to_token) do 
    vocab_size = vocab_size + 1 
end
vocab_size=vocab_size+1  --other strange symbols 
print('vocab_size: '.. vocab_size)
opt.vocab_size = vocab_size

--- Set up the model
local dtype = 'torch.CudaTensor'
print('initializing SAR model from scratch...')
local model = SARModel(opt):type(dtype)
model:convert(dtype, 1)

-- get the parameters vector
local cnn_params, grad_cnn_params, rec_params, grad_rec_params
if opt.init_from == '' then
  cnn_params, grad_cnn_params, rec_params, grad_rec_params = model:getParameters()
  print('total number of parameters in CNN: ', cnn_params:nElement())
  print('total number of parameters in rec_net: ', rec_params:nElement())

else
  local checkpoint =torch.load(opt.init_from)
  model = checkpoint.model
  model:type(dtype)
  cnn_params, grad_cnn_params, rec_params, grad_rec_params = model:getParameters()

end

-- Initialize training information
local loss_history = {}
local val_losshistory ={}
local iter = 1
local cnn_optim_state = {}
local rec_optim_state = {}
model:clearState()

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  grad_rec_params:zero()   
  if opt.finetune_cnn_after ~= -1 and iter >= opt.finetune_cnn_after then
    grad_cnn_params:zero() 
  end
  model:training()

  -- loading data
  local data = {}
  data.image, data.gt_labels, data.image_W = trainSet:nextBatch(opt)
  for k, v in pairs(data) do
    data[k] = v:type(dtype)
  end
  
  -- Run the model forward and backward
  model.cnn_backward = false
  if opt.finetune_cnn_after ~= -1 and iter > opt.finetune_cnn_after then
    model.finetune_cnn = true
  end
  local losses, stats = model:forward_backward(data)

  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- Logging code
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  if opt.losses_log_every > 0 and iter % opt.losses_log_every == 0 then
    loss_history[iter] = losses
  end

  return losses, stats
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
while true do  
  model:clearState()
  -- Compute loss and gradient
  local losses, stats = lossFun()
  
  -- Parameter update
  adam(rec_params, grad_rec_params, opt.learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, rec_optim_state)
  -- Make a step on the CNN if finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    adam(cnn_params, grad_cnn_params, opt.learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
  end

  -- print loss and timing/benchmarks
  print(string.format('iter %d: batch training loss %s', iter, losses))
  if (iter > 0 and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) then
    local valdata = {}
    valdata.image, valdata.gt_labels,valdata.image_W = valSet:allImages(opt)
    for k, v in pairs(valdata) do
      valdata[k] = v:type(dtype)
    end
    local nFrame = valdata.image:size(1)
    local val_losses = {}

    for i = 1, nFrame, opt.batch_size do
      local actualBatchSize = math.min(opt.batch_size, nFrame-i+1)
      local inputBatch = valdata.image:narrow(1,i,actualBatchSize)
      local inputBatch_W = valdata.image_W:narrow(1,i,actualBatchSize)
      local outputBatch = valdata.gt_labels:narrow(1,i,actualBatchSize)

      local lossBatch, captionBatch = model:forward_val(inputBatch, inputBatch_W, outputBatch)
      table.insert(val_losses,lossBatch)
    end
  
    local alllosses=torch.Tensor(val_losses)
    local val_loss = torch.mean(alllosses)
    print('validation loss: ', val_loss)
    val_losshistory[iter] = val_loss

    -- serialize a json file that has all info except the model
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_losshistory = val_losshistory
    cjson.encode_number_precision(4) -- number of sig digits to use in encoding
    cjson.encode_sparse_array(true, 2, 10)
    local text = cjson.encode(checkpoint) 
    local file_name = string.format('checkpoint_epoch%d_%.4f', iter, val_loss)
    local save_path = paths.concat(opt.checkpoint_path,file_name)
    local file = io.open(save_path .. '.json', 'w')
    file:write(text)
    file:close()
    print('wrote accompany data to' .. save_path .. '.json')

    -- save the model
    checkpoint.model = model
    model:clearState()

    local net_path = save_path .. '.t7'
    torch.save(net_path, checkpoint)
    print('wrote model to' .. opt.checkpoint_path)

    cnn_params, grad_cnn_params, rec_params, grad_rec_params = model:getParameters()
  end
  
  -- changing training data groups...
  if (iter > 0 and iter % opt.change_train_every == 0) then
    ij=ij+1
    opt.trainSetPath='DataDB/train_' .. tostring(ij) .. '/data.mdb'
    print(opt.trainSetPath)
    trainSet = DatasetLmdb(opt.trainSetPath, opt.batch_size)
  end
  
  -- stopping criterions
  iter = iter + 1
  -- Collect garbage every so often
  if iter % 33 == 0 then collectgarbage() end
  if iter % opt.learning_rate_decay_every == 0 then 
    print('learning rate' .. opt.learning_rate)
    if opt.learning_rate > 1e-5 then
      local decay_factor = opt.learning_rate_decay
      opt.learning_rate = opt.learning_rate * decay_factor -- decay it
      print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. opt.learning_rate)
    end
  end
  if loss0 == nil then loss0 = losses end
  if losses > loss0 * 100 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end
end
