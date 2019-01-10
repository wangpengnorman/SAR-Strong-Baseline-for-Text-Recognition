local pathCache = package.path
package.path = 'third_party/lmdb-lua-ffi/src/?.lua'
local lmdb = require('lmdb')
package.path = pathCache
local Image = require('image')
require('utilities')

local DatasetLmdb = torch.class('DatasetLmdb')

function DatasetLmdb:__init(lmdbPath, batchSize, imageType)
    self.batchSize = batchSize or -1
    self.imageType = imageType or 'jpg'
    self:loadDataset(lmdbPath)
end


function DatasetLmdb:loadDataset(lmdbPath)

    self.env = lmdb.environment(lmdbPath, {subdir=false, max_dbs=8, size=1099511627776})
    self.env:transaction(function(txn)
        self.nSamples = tonumber(tostring(txn:get('num-samples')))
    end)
end


function DatasetLmdb:getNumSamples()
    return self.nSamples
end

function DatasetLmdb:nextBatch(opt)
    local imgW, imgH = 160, 48
    local imgW_min = 48

    local randomIndex = torch.LongTensor(self.batchSize):random(1, self.nSamples)
    local imageList, labelList = {}, {}

    -- load image binaries and labels
    local success, msg, rc = self.env:transaction(function(txn)
        for i = 1, self.batchSize do
            local idx = randomIndex[i]
            local imageKey = string.format('image-%09d', idx)
            local labelKey = string.format('label-%09d', idx)
            local imageBin = txn:get(imageKey)
            local labelBin = txn:get(labelKey)
            imageList[i] = tostring(imageBin)
            labelList[i] = tostring(labelBin)
        end
    end)

    -- decode images
    local images = torch.ByteTensor(self.batchSize, 3, imgH, imgW):fill(0)
    local images_W = torch.ByteTensor(self.batchSize, 1)
    for i = 1, self.batchSize do
        local imgBin = imageList[i]
        local img0 = Image.load(imgBin,3,'byte')
        local ratn = math.random(0,4)   
        local img
        if ratn == 0 then   --rotation for data augmentation
            local deg = math.random(-15, 15)
            img = Image.rotate(img0, deg * math.pi / 180)
        else
            img = img0
        end

        local ow,oh = img:size(3), img:size(2)
        local W=torch.round(ow*imgH/oh)
        if W>imgW then
            img = Image.scale(img, imgW, imgH)
            images:sub(i,i,1,-1,1,-1,1,imgW):copy(img)
            images_W[i]=imgW
        elseif W>imgW_min and W<=imgW then
            img = Image.scale(img, W, imgH)
            images:sub(i,i,1,-1,1,-1,1,W):copy(img)
            images_W[i]=W
        elseif W<=imgW_min then
            img = Image.scale(img, imgW_min, imgH)
            images:sub(i,i,1,-1,1,-1,1,imgW_min):copy(img)
            images_W[i]=imgW_min
        end
    end

    local labels = str2label(labelList, opt.token_to_idx, opt.vocab_size, opt.seq_length)

    collectgarbage()
    return images, labels, images_W
end


function DatasetLmdb:allImages(opt, nSampleMax)
    local imgW, imgH = 160, 48
    local imgW_min = 48
    nSampleMax = nSampleMax or math.huge
    local nSample = math.min(self.nSamples, nSampleMax)
    local images = torch.ByteTensor(nSample, 3, imgH, imgW):fill(0)
    local images_W = torch.ByteTensor(nSample, 1)
    local labelList = {}
    self.env:transaction(function(txn)
        for i = 1, nSample do
            local imageKey = string.format('image-%09d', i)
            local labelKey = string.format('label-%09d', i)
            local imageBin = tostring(txn:get(imageKey))
            local labelBin = tostring(txn:get(labelKey))
            local imageByteLen = string.len(imageBin)
            local imageBytes = torch.ByteTensor(imageByteLen)
            imageBytes:storage():string(imageBin)
            local img = Image.decompress(imageBytes, 3, 'byte')
            local ow,oh = img:size(3), img:size(2)
            local W=torch.round(ow*imgH/oh)
            if W>imgW then
                img = Image.scale(img, imgW, imgH)
                images:sub(i,i,1,-1,1,-1,1,imgW):copy(img)
                images_W[i]=imgW
            elseif W>imgW_min and W<=imgW then
                img = Image.scale(img, W, imgH)
                images:sub(i,i,1,-1,1,-1,1,W):copy(img)
                images_W[i]=W
            elseif W<=imgW_min then
                img = Image.scale(img, imgW_min, imgH)
                images:sub(i,i,1,-1,1,-1,1,imgW_min):copy(img)
                images_W[i]=imgW_min
            end
            labelList[i] = labelBin
        end
    end)

    local labels = str2label(labelList, opt.token_to_idx, opt.vocab_size, opt.seq_length)
    collectgarbage()
    return images, labels, images_W
end

function DatasetLmdb:getImages(opt, i)
    local imgW, imgH = 160, 48
    local imgW_min = 48
    local images 
    local images_W 
    self.env:transaction(function(txn)
        local imageKey = string.format('image-%09d', i)
        local imageBin = tostring(txn:get(imageKey))
        local imageByteLen = string.len(imageBin)
        local imageBytes = torch.ByteTensor(imageByteLen)
        imageBytes:storage():string(imageBin)
        local img = Image.decompress(imageBytes, 3, 'byte')
        local ow,oh = img:size(3), img:size(2)

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
    end)

    collectgarbage()
    return images, images_W
end
