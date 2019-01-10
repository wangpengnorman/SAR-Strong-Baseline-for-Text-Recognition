require 'torch'
require 'nn'
require 'nngraph'
local Image = require 'image'

require 'recognition_net'
require 'TemporalCrossEntropyCriterion'

local SARModel, parent = torch.class('SARModel', 'nn.Module')

function SARModel:__init(opt)

  opt = opt or {}  
  opt.backend = getopt(opt, 'backend', 'cudnn')
  opt.dtype = getopt(opt, 'dtype', 'torch.CudaTensor')
  opt.vocab_size = getopt(opt, 'vocab_size')
  opt.std = getopt(opt, 'std', 0.01) -- Used to initialize new layers
  
  -- Options for RNN
  opt.seq_length = getopt(opt, 'seq_length')
  opt.rnn_size = getopt(opt, 'rnn_size', 512)
  opt.input_encoding_size = getopt(opt, 'input_encoding_size', 512)
  opt.encoder_size = getopt(opt, 'encoder_size', 512)
  opt.batch_size = getopt(opt, 'batch_size', 128)
  opt.max_height, opt.max_width = 6, 40

  self.opt = opt 
  -- This will hold various components of the model
  self.nets = {}
  
  self.nets.conv_net = self:CNNNet()
  
  -- Set up parameters for recognizer
  local lm_opt = {
    vocab_size = opt.vocab_size,
    input_encoding_size = opt.input_encoding_size,
    rnn_size = opt.rnn_size,
    seq_length = opt.seq_length,
    idx_to_token = opt.idx_to_token,
    max_height=opt.max_height, 
    max_width=opt.max_width,
    batch_size=opt.batch_size,
    num_layers=opt.num_layers,
    beam_size = opt.beam_size,
  }

  self.nets.recog_net = nn.recog_net(lm_opt)

  -- Set up Criterions
  self.crits = nn.TemporalCrossEntropyCriterionme()

  self:training()
  self.finetune_cnn = false

end

-- ResNet based CNN architecture
function SARModel:CNNNet()

    local Convolution = cudnn.SpatialConvolution
    local Avg = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local Max = nn.SpatialMaxPooling
    local SBatchNorm = nn.SpatialBatchNormalization

    local iChannels
      -- The shortcut layer is either identity or 1x1 convolution
    local function shortcut(nInputPlane, nOutputPlane, stride)
      if nInputPlane ~= nOutputPlane then
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      else
         return nn.Identity()
      end
    end


    local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
    end

    -- Creates count residual blocks with specified number of features
    local function layer(block, features, count, stride)
      local iden = 0
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, 1))
      end
      return s
    end

    local model = nn.Sequential()
   
    iChannels = 128
    print(' | ResNet- text')

    -- The ResNet model
    model:add(nn.AddConstant(-128.0))
    model:add(Convolution(3,64,3,3,1,1,1,1))
    model:add(SBatchNorm(64))
    model:add(ReLU(true))
    model:add(Convolution(64,128,3,3,1,1,1,1))
    model:add(SBatchNorm(128))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(layer(basicblock, 256, 1))
    model:add(Convolution(256,256,3,3,1,1,1,1))
    model:add(SBatchNorm(256))
    model:add(ReLU(true))
    model:add(Max(2, 2, 2, 2))
    model:add(layer(basicblock, 256, 2))
    model:add(Convolution(256,256,3,3,1,1,1,1))
    model:add(SBatchNorm(256))
    model:add(ReLU(true))
    model:add(Max(1, 2, 1, 2))
    model:add(layer(basicblock, 512, 5))
    model:add(Convolution(512,512,3,3,1,1,1,1))
    model:add(SBatchNorm(512))
    model:add(ReLU(true))
    -- model:add(Max(1, 2, 1, 2))
    model:add(layer(basicblock, 512, 3))
    model:add(Convolution(512,512,3,3,1,1,1,1))
    model:add(SBatchNorm(512))
    model:add(ReLU(true))

    -- local function ConvInit(name)
    --   for k,v in pairs(model:findModules(name)) do
    --       local n = v.kW*v.kH*v.nOutputPlane
    --       v.weight:normal(0,math.sqrt(2/n))
    --       if cudnn.version >= 4000 then
    --         v.bias = nil
    --         v.gradBias = nil
    --       else
    --         v.bias:zero()
    --       end
    --   end
    -- end

    -- local function BNInit(name)
    --   for k,v in pairs(model:findModules(name)) do
    --       v.weight:fill(1)
    --       v.bias:zero()
    --   end
    -- end

    -- ConvInit('cudnn.SpatialConvolution')
    -- ConvInit('nn.SpatialConvolution')
    -- BNInit('cudnn.SpatialBatchNormalization')
    -- BNInit('nn.SpatialBatchNormalization')

    -- -- model:get(1).gradInput = nil

    return model
end
 
function SARModel:training()
  parent.training(self)
  self.nets.conv_net:training()
  self.nets.recog_net:training()
end

function SARModel:evaluate()
  parent.evaluate(self)
  self.nets.conv_net:evaluate()
  self.nets.recog_net:evaluate()
end

--[[
Convert this SARModel to a particular datatype, and convert convolutions
between cudnn and nn.
--]]
function SARModel:convert(dtype, use_cudnn)
  self:type(dtype)
  if cudnn and use_cudnn ~= nil then
    local backend = nn
    if use_cudnn then
      backend = cudnn
    end
    cudnn.convert(self.nets.conv_net, backend)
    cudnn.convert(self.nets.recog_net, backend)
  end
end


--[[Run the model forward.]]
function SARModel:updateOutput(input)
  assert(input:dim() == 4 and input:size(2) == 3)
  H, W = input:size(3), input:size(4)

  if self.train then
    self.cnn_features = self.nets.conv_net:forward(input)
    self.featWidth = {}
    self.cnn_features_mask=torch.Tensor(self.opt.batch_size, self.opt.max_height, self.opt.max_width):fill(1):cuda()

    for i =1, self.opt.batch_size do
      local fw = torch.floor(self.image_size[i]/4)
      table.insert(self.featWidth,fw[1])
      self.cnn_features_mask:sub(i,i,1,-1,1,fw[1]):fill(0)
      if fw[1]<self.opt.max_width then
        self.cnn_features:sub(i,i,1,-1,1,-1,fw[1]+1,-1):fill(0)
      end
    end
    self.featWidth=torch.Tensor(self.featWidth):cuda()
    self.output = self.nets.recog_net:forward{self.cnn_features, self.target_labels, self.cnn_features_mask, self.featWidth}
  end

  if not self.train then
    local actualbatch = input:size(1)
    local cnn_features= self.nets.conv_net:forward(input)
    local featWidth = {}
    local cnn_features_mask=torch.Tensor(actualbatch, self.opt.max_height, self.opt.max_width):fill(1):cuda()
    
    for i =1, actualbatch do
      local fw = torch.floor(self.image_size[i]/4)
      table.insert(featWidth,fw[1])
      cnn_features_mask:sub(i,i,1,-1,1,fw[1]):fill(0)
      if fw[1]<self.opt.max_width then
        cnn_features:sub(i,i,1,-1,1,-1,fw[1]+1,-1):fill(0)
      end
    end
    featWidth=torch.Tensor(featWidth):cuda()
    local empty = cnn_features.new()
    self.output = self.nets.recog_net:forward{cnn_features, empty, cnn_features_mask, featWidth}
  end 
  return self.output
end


function SARModel:backward(input, gradOutput)

  local dout_recogR = self.nets.recog_net:backward({self.cnn_features, self.target_labels, self.cnn_features_mask, self.featWidth}, gradOutput)
  local dout_recogR1 = dout_recogR[1]  
  local dout_cnn = self.nets.conv_net:backward(input, dout_recogR1)

  self.gradInput = dout_cnn
  return self.gradInput
end


function SARModel:forward_test(input, image_size)
  self:evaluate()
  self.image_size = image_size

  local cap_out = self:forward(input)

  local pre_output = cap_out[1]
  local seq = cap_out[2]
  local seqScore = cap_out[3]
  local attW = cap_out[4]:double()

  local captions
  if seq:nDimension()>0 then
    captions, captions_sco = self.nets.recog_net:decodeSequence(seq, seqScore)
  else
    captions = {}
    captions_sco = {}
  end
  return captions, captions_sco
end

function SARModel:forward_val(imag, image_size, target_labels)
  self:evaluate()
  self.image_size = image_size

  local cap_out = self:forward(imag)
  local pre_output = cap_out[1]
  local seq = cap_out[2]
  local seqScore = cap_out[3]
  local attW = cap_out[4]:double()


  local target = self.nets.recog_net:getTarget(target_labels)
  local losses = self.crits:forward(pre_output, target)
  local captions = self.nets.recog_net:decodeSequence_org(seq)
  local captions_gt = self.nets.recog_net:decodeSequence_org(target_labels)
  print(captions, captions_gt)
  -- debug.debug()

  return losses, captions
end


function SARModel:forward_backward(data)
  self:training()

  self.target_labels = data.gt_labels
  self.image_size = data.image_W
  local lm_output = self:forward(data.image)
  -- Compute captioning loss
  local target = self.nets.recog_net:getTarget(self.target_labels)
  local losses = self.crits:forward(lm_output, target)
  -- print('Training loss', losses)
  local grad_lm_output = self.crits:backward(lm_output, target)

  self:backward(data.image, grad_lm_output)

  return losses
end

function SARModel:getParameters()
  local cnn_params, grad_cnn_params = self.nets.conv_net:getParameters()
  local rec_params, grad_rec_params = self.nets.recog_net:getParameters()
  return cnn_params, grad_cnn_params, rec_params, grad_rec_params
end


function SARModel:clearState()
  self.nets.conv_net:clearState()
  self.nets.recog_net:clearState()
  self.crits:clearState()
end


