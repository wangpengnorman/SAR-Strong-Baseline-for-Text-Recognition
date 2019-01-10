require 'loadcaffe'


local net_utils = {}


function net_utils.load_cnn(name, backend, path_offset)
  local model_dir, proto_file, model_file = nil, nil, nil
  if name == 'vgg-16' then
    model_dir = 'data/models/vgg-16'
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
  return net_utils.cudnn_tune_cnn(name, cnn)
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
--[[
-- These seeem to use too much memory =(
cudnn_algos['vgg-16']['GeForce GTX TITAN Z'] = {
  [1] = {0, 0, 1},
  [3] = {2, 1, 1},
  [6] = {1, 0, 3},
  [8] = {1, 1, 3},
  [11] = {1, 0, 0},
  [13] = {1, 0, 0},
  [15] = {1, 0, 0},
  [18] = {1, 0, 0},
  [20] = {1, 0, 0},
  [22] = {1, 0, 0},
  [25] = {1, 0, 3},
  [27] = {1, 0, 3},
  [29] = {1, 0, 3},
}
--]]


return net_utils
