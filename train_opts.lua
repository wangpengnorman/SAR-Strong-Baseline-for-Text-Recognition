local M = { }

-- parameters used in the code
function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a word recognition model')
  cmd:text()
  cmd:text('Options')
  -- data
  -- cmd:option('-trainSetPath','DataDB/comb_train2/data.mdb','training data path')
  cmd:option('-valSetPath','DataDB/IC13_test/data.mdb','test data path')
  cmd:option('-token_dir','DataDB/token_dir','test data path')
  
  -- model params
  cmd:option('-rnn_size', 512, 'size of LSTM internal state')
  cmd:option('-input_encoding_size', 512, 'Dimension of the word vectors to use in the RNN')
  cmd:option('-num_layers', 2, 'number of layers in the LSTM')
  cmd:option('-model', 'lstm', 'lstm, gru or rnn')
  cmd:option('-seq_length',30,'number of timesteps to unroll for')
  cmd:option('-batch_size',18,'number of sequences to train on in parallel')
  cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
  cmd:option('-max_iters', 3000000, 'Number of iterations to run; -1 to run forever')
  cmd:option('-finetune_cnn_after', 0,
    'Start finetuning CNN after this many iterations (-1 = never finetune)')

  -- optimization
  cmd:option('-optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
  cmd:option('-learning_rate',8e-4,'learning rate')
  cmd:option('-learning_rate_decay',0.9,'learning rate decay')
  cmd:option('-learning_rate_decay_every',10000,'in number of iteration, decaying the learning rate')
  cmd:option('-optim_alpha', 0.9, 'alpha for adagrad/rmsprop/momentum/adam')
  cmd:option('-optim_beta', 0.999, 'beta used for adam')
  cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')
  cmd:option('-cnn_optim','adam', 'optimization to use for CNN')
  cmd:option('-cnn_optim_alpha', 0.9,' alpha for momentum of CNN')
  cmd:option('-cnn_optim_beta', 0.999, 'alpha for momentum of CNN')
  cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for the CNN')
  cmd:option('-grad_clip',5,'clip gradients at this value')
 
  -- bookkeeping
  cmd:option('-checkpoint_path', 'saved_model', 'output directory where checkpoints get written')
  cmd:option('-save_checkpoint_every', 10,'How often to save model checkpoints')
  cmd:option('-change_train_every', 20000,'How often to save model checkpoints')
  cmd:option('-losses_log_every', 10,
    'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

  -- GPU/CPU
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-gpu',3,'which gpu to use. -1 = use CPU')

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
