local Image = require('image')
local cjson = require 'cjson'
require 'loadcaffe'

function str2label(strs, token_to_idx, vocab_size, seq_length)
    --[[ Convert a list of strings to integer label tensor (zero-padded).

    ARGS:
      - `strs`     : table, list of strings
      - `maxLength`: int, the second dimension of output label tensor

    RETURN:
      - `labels`   : tensor of shape [#(strs) x maxLength]
    ]]
    assert(type(strs) == 'table')
    local nStrings = #strs
    local labels = torch.IntTensor(nStrings, seq_length):fill(0)
    for i, str in ipairs(strs) do
        if string.len(str)<=seq_length then
            for j = 1, string.len(str) do
                char = string.sub(str,j,j) 
                if token_to_idx[char] then
                    labels[i][j] = token_to_idx[char]
                else 
                    labels[i][j] = vocab_size
                end
            end
        else
            -- print(str)
            -- debug.debug()
            for j = 1, seq_length do
                char = string.sub(str,j,j) 
                if token_to_idx[char] then
                    labels[i][j] = token_to_idx[char]
                else 
                    labels[i][j] = vocab_size
                end
            end
        end

    end
    return labels
end


function label2str(labels, raw)
    --[[ Convert a label tensor to a list of strings.

    ARGS:
      - `labels`: int tensor, labels
      - `raw`   : boolean, if true, convert zeros to '-'

    RETURN:
      - `strs`  : table, list of strings
    ]]
    assert(labels:dim() == 2)
    raw = raw or false

    function label2ascii(label)
        local ascii
        if label >= 1 and label <= 10 then
            ascii = label - 1 + 48
        elseif label >= 11 and label <= 36 then
            ascii = label - 11 + 97
        elseif label == 0 then -- used when displaying raw predictions
            ascii = string.byte('-')
        end
        return ascii
    end

    local strs = {}
    local nStrings, maxLength = labels:size(1), labels:size(2)
    for i = 1, nStrings do
        local str = {}
        local labels_i = labels[i]
        for j = 1, maxLength do
            if raw then
                str[j] = label2ascii(labels_i[j])
            else
                if labels_i[j] == 0 then
                    break
                else
                    str[j] = label2ascii(labels_i[j])
                end
            end
        end
        str = string.char(unpack(str))
        strs[i] = str
    end
    return strs
end

function cloneList(tensors, fillZero)
    --[[ Clone a list of tensors, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `tensors`  : table, list of tensors
      - `fillZero` : boolean, if true tensors are filled with zeros
    RETURNS:
      - `output`   : table, cloned list of tensors
    ]]
    local output = {}
    for k, v in pairs(tensors) do
        output[k] = v:clone()
        if fillZero then output[k]:zero() end
    end
    return output
end


function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end



function modelState(model)
    --[[ Get model state, including model parameters (weights and biases) and
         running mean/var in batch normalization layers
    ARGS:
      - `model` : network model
    RETURN:
      - `state` : table, model states
    ]]
    local parameters = model:parameters()
    local bnVars = {}
    local bnLayers = model:findModules('cudnn.BatchNormalization')
    for i = 1, #bnLayers do
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end
    local bnLayers = model:findModules('cudnn.SpatialBatchNormalization')
    for i = 1, #bnLayers do
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end
    local state = {parameters = parameters, bnVars = bnVars}
    return state
end

function loadModelState(model, stateToLoad)
    local state = modelState(model)
    state.parameters[1]:sub(1,-1,1,1,1,-1,1,-1):copy(stateToLoad.parameters[1])
    state.parameters[1]:sub(1,-1,2,2,1,-1,1,-1):copy(stateToLoad.parameters[1])
    state.parameters[1]:sub(1,-1,3,3,1,-1,1,-1):copy(stateToLoad.parameters[1])
    for i = 2, #state.parameters do
        state.parameters[i]:copy(stateToLoad.parameters[i])
    end
    for i = 2, #state.bnVars-2 do
        state.bnVars[i]:copy(stateToLoad.bnVars[i])
    end
end



function loadModelState_nonorm(model, stateToLoad)
    local state = modelState(model)
    -- print(#state.parameters)
    -- print(#stateToLoad.parameters)
    -- debug.debug()
    -- for i = 1, #state.parameters do
    --     print(state.parameters[i]:size())
    --     print(stateToLoad.parameters[i]:size())
    -- end
    -- print(stateToLoad.parameters[13]:size())
    -- print(stateToLoad.parameters[14]:size())
    -- print(stateToLoad.parameters[15]:size())
    -- print(stateToLoad.parameters[16]:size())
    -- debug.debug()

    for i = 1, 6 do
        state.parameters[i]:copy(stateToLoad.parameters[i])
    end
    state.parameters[7]:copy(stateToLoad.parameters[9])
    state.parameters[8]:copy(stateToLoad.parameters[10])
    state.parameters[9]:copy(stateToLoad.parameters[11])
    state.parameters[10]:copy(stateToLoad.parameters[12])
    state.parameters[11]:copy(stateToLoad.parameters[15])
    state.parameters[12]:copy(stateToLoad.parameters[16])
    -- for i = 1, #state.bnVars do
    --     state.bnVars[i]:copy(stateToLoad.bnVars[i])
    -- end
end



-- Assume required if default_value is nil
function getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end



-- Count the time of a function
function timeit(f)
  timer = timer or torch.Timer()
  cutorch.synchronize()
  timer:reset()
  f()
  cutorch.synchronize()
  return timer:time().real
end

function write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

function image_rotate(im, augRot)
   
    a = augRot * math.pi / 180
    s = math.sin(a)
    c = math.cos(a)
    h = im:size(2)
    w = im:size(3)
    hh = math.floor(h*(c+s+2)/2)
    ww = math.floor(w*(c+s)/2)

    eg =im[{{1},{1},{1}}]:squeeze()
    src = torch.zeros(3, h*(s+c+4), w*(s+c+1)):fill(eg):byte()
    -- print(src:size(),im:size(),hh,ww,h,w)
    if hh == 0 or ww == 0 or hh+h-1>src:size(2) or ww+w-1>src:size(3) then
        res_im=im
    else

        src[{{},{hh,hh+h-1},{ww,ww+w-1}}] = im
        -- Image.save('tmp3.png',src)

        dst = image.rotate(src, a)
        -- Image.save('tmp4.png',dst)
        idx = dst:nonzero()
        testidx = idx:sub(1,-1,3,3):squeeze()
        idr, _ = torch.max(testidx,1)
        idl, _ = torch.min(testidx,1)
        res_im = dst:sub(1,-1,1,-1,idl[1],idr[1])
    end

    return res_im

end

function subsequence(net, start_idx, end_idx)
  local seq = nn.Sequential()
  for i = start_idx, end_idx do
    seq:add(net:get(i))
  end
  return seq
end

function load_cnn(name, backend, path_offset)
  local model_dir, proto_file, model_file = nil, nil, nil
  if name == 'vgg-16' then
    model_dir = 'model/vgg-16'
    proto_file = 'VGG_ILSVRC_16_layers_deploy.prototxt'
    model_file = 'VGG_ILSVRC_16_layers.caffemodel'
  else
    error(string.format('Unrecognized model "%s"', name))
  end
  if path_offset then
    model_dir = paths.concat(path_offset, model_dir)
  end
  print('loading network weights from .. ' .. model_file)
  proto_file = paths.concat(model_dir, proto_file)
  model_file = paths.concat(model_dir, model_file)
  local cnn = loadcaffe.load(proto_file, model_file, backend)
  return cudnn_tune_cnn(name, cnn)
end


-- Hardcode good cudnn v3 algorithms for different networks and GPUs.
-- We can't just run cudnn in benchmark mode because it will recompute
-- benchmarks for every new image size, which will be very slow; instead
-- we just pick some good algorithms for large images (800 x 600). They
-- might not be optimal for all image sizes but will probably be better
-- than the defaults.
local cudnn_algos = {}
cudnn_algos['vgg-16'] = {}
cudnn_algos['vgg-16']['GeForce GTX TITAN X'] = {
  [1] = {1, 0, 0},
  [3] = {1, 1, 0},
  [6] = {1, 1, 3},
  [8] = {1, 1, 3},
  [11] = {1, 1, 3},
  [13] = {1, 1, 3},
  [15] = {1, 1, 3},
  [18] = {1, 1, 3},
  [20] = {1, 1, 0},
  [22] = {1, 1, 0},
  [25] = {1, 1, 0},
  [27] = {1, 1, 0},
  [29] = {1, 1, 0},
}

function cudnn_tune_cnn(cnn_name, cnn)
  if not cutorch then
    return cnn
  end
  local device = cutorch.getDevice()
  local device_name = cutorch.getDeviceProperties(device).name
  if cudnn_algos[cnn_name] and cudnn_algos[cnn_name][device_name] then
    local algos = cudnn_algos[cnn_name][device_name]
    for i = 1, #cnn do
      local layer = cnn:get(i)
      if torch.isTypeOf(layer, 'cudnn.SpatialConvolution') and algos[i] then
        layer:setMode(unpack(algos[i]))
      end
    end
  end
  return cnn
end