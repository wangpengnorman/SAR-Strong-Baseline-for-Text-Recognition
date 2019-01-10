require('torch')
require('nn')
require('paths')
require('nngraph')
local Image = require('image')
require('utilities')
require('DatasetLmdb')
require('MainModel_recog')
require('hdf5')

--Run a trained recognition model on images.
local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint', 'saved_model/BestModel.t7')
cmd:option('-input_image', '', 'A path to a single specific image to caption')
cmd:option('-input_dir', '', 'A path to a directory with images to caption')
cmd:option('-token_dir','DataDB/token_dir','test data path')
cmd:option('-output_vis_dir', 'vis')
cmd:option('-seq_length',30,'number of timesteps to unroll for')
cmd:option('-max_images', 1000, 'max number of images to process')
cmd:option('-beam_size', 5, 'use beam search or not')
-- Misc
cmd:option('-gpu', 3)
cmd:option('-use_cudnn', 1)
cmd:option('-seed',123,'torch manual random number generator seed')
local opt = cmd:parse(arg)


function loadAndResizeImage(imagePath)
    local imgW, imgH = 160, 48
    local imgW_min = 48
    local images 
    local images_W 
    local img = Image.load(imagePath, 3, 'byte')
    local ow,oh = img:size(3), img:size(2)
        --  if image width is smaller than image height, rotate the image 90 degree clockwise and anticlockwise respectively.
        if oh > 1.1*ow then
            images = torch.ByteTensor(3, 3, imgH, imgW):fill(0)
            images_W = torch.ByteTensor(3, 1) 
            local img1=torch.ByteTensor(3, ow, oh):fill(0)
            for j = 1, ow do
                img1[{{},{j},{}}]=img[{{},{},{j}}]:transpose(2,3):index(3, torch.linspace(oh,1,oh):long())
            end
            local ow1,oh1 = img1:size(3), img1:size(2)

            local W=torch.round(ow1*imgH/oh1)
            if W>imgW then
                img1 = Image.scale(img1, imgW, imgH)
                images:sub(1,1,1,-1,1,-1,1,imgW):copy(img1)
                images_W[1]=imgW
            elseif W>imgW_min and W<=imgW then
                img1 = Image.scale(img1, W, imgH)
                images:sub(1,1,1,-1,1,-1,1,W):copy(img1)
                images_W[1]=W
            elseif W<=imgW_min then
                img1 = Image.scale(img1, imgW_min, imgH)
                images:sub(1,1,1,-1,1,-1,1,imgW_min):copy(img1)
                images_W[1]=imgW_min
            end

            local img2=torch.ByteTensor(3, ow, oh):fill(0)
            for j = 1, ow do
                img2[{{},{j},{}}]=img[{{},{},{ow-j+1}}]:transpose(2,3)
            end
            local ow2,oh2 = img2:size(3), img2:size(2)

            local W=torch.round(ow2*imgH/oh2)
            if W>imgW then
                img2 = Image.scale(img2, imgW, imgH)
                images:sub(2,2,1,-1,1,-1,1,imgW):copy(img2)
                images_W[2]=imgW
            elseif W>imgW_min and W<=imgW then
                img2 = Image.scale(img2, W, imgH)
                images:sub(2,2,1,-1,1,-1,1,W):copy(img2)
                images_W[2]=W
            elseif W<=imgW_min then
                img2 = Image.scale(img2, imgW_min, imgH)
                images:sub(2,2,1,-1,1,-1,1,imgW_min):copy(img2)
                images_W[2]=imgW_min
            end

            local W=torch.round(ow*imgH/oh)
            if W>imgW then
                img = Image.scale(img, imgW, imgH)
                images:sub(3,3,1,-1,1,-1,1,imgW):copy(img)
                images_W[3]=imgW
            elseif W>imgW_min and W<=imgW then
                img = Image.scale(img, W, imgH)
                images:sub(3,3,1,-1,1,-1,1,W):copy(img)
                images_W[3]=W
            elseif W<=imgW_min then
                img = Image.scale(img, imgW_min, imgH)
                images:sub(3,3,1,-1,1,-1,1,imgW_min):copy(img)
                images_W[3]=imgW_min
            end

        else
          -- if image width is larger than its height, resize the original image directly.
            images = torch.ByteTensor(1, 3, imgH, imgW):fill(0)
            images_W = torch.ByteTensor(1, 1) 
        
            local W=torch.round(ow*imgH/oh)
            if W>imgW then
                img = Image.scale(img, imgW, imgH)
                images:sub(1,1,1,-1,1,-1,1,imgW):copy(img)
                images_W[1]=imgW
            elseif W>imgW_min and W<=imgW then
                img = Image.scale(img, W, imgH)
                images:sub(1,1,1,-1,1,-1,1,W):copy(img)
                images_W[1]=W
            elseif W<=imgW_min then
                img = Image.scale(img, imgW_min, imgH)
                images:sub(1,1,1,-1,1,-1,1,imgW_min):copy(img)
                images_W[1]=imgW_min
            end
        end

    collectgarbage()
    return images, images_W
end

function get_input_images(opt)
  -- utility function that figures out which images we should process 
  -- and fetches all the raw image paths
  local image_paths = {}
  if opt.input_image ~= '' then
    table.insert(image_paths, opt.input_image)
  elseif opt.input_dir ~= '' then
    -- iterate all files in input directory and add them to work
    for fn in paths.files(opt.input_dir) do
      if string.sub(fn, 1, 1) ~= '.' then
        local img_in_path = paths.concat(opt.input_dir, fn)
        table.insert(image_paths, img_in_path)
      end
    end
  else
    error('one of input_image, input_dir, or input_split must be provided.')
  end
  return image_paths
end


-- Load the model, and cast to the right type
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  -- cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
end

local dtype = 'torch.CudaTensor'
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

model:type(dtype)
model:evaluate()

--get the all the character tokens
local vocab_file = path.join(opt.token_dir, 'vocab.t7')
local token_file = path.join(opt.token_dir, 'token.t7')
idx_to_token = torch.load(token_file)
token_to_idx = torch.load(vocab_file)
-- count vocab
vocab_size = 0
for _ in pairs(idx_to_token) do 
    vocab_size = vocab_size + 1 
end
vocab_size=vocab_size+1

opt.vocab_size = vocab_size
opt.idx_to_token = idx_to_token
opt.token_to_idx = token_to_idx

--get the test image path
local image_paths = get_input_images(opt)
local num_process = math.min(#image_paths, opt.max_images)
-- get the test image
local captions = {}
local results_json = {}
for k=1,num_process do
    local img_path = image_paths[k]
    print(string.format('%d/%d processing image %s', k, num_process, img_path))
    local image, image_W = loadAndResizeImage(img_path)
    image = image:type(dtype)
    image_W =image_W:type(dtype)
    local cap_im, cap_score = model:forward_test(image, image_W)
    local result_json={}
    if #cap_im>1 then
        tmpcap, tmpsco = torch.max(torch.Tensor(cap_score),1)
        result_json.caption=cap_im[tmpsco[1]]
    else
        result_json.caption=cap_im[1]
    end
    local tmpid=paths.basename(img_path)
    result_json.img_id = tmpid
    table.insert(results_json, result_json)
end
print(results_json)

write_json(paths.concat(opt.output_vis_dir, 'result.json'), results_json)
