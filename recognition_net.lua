require 'nn'
require 'maskSoftMax'

local recog, parent = torch.class('nn.recog_net', 'nn.Module')

function recog:__init(opt)
  parent.__init(self)

  opt = opt or {}
  self.vocab_size = getopt(opt, 'vocab_size')
  self.input_encoding_size = getopt(opt, 'input_encoding_size')
  self.rnn_size = getopt(opt, 'rnn_size')
  self.seq_length = getopt(opt, 'seq_length')
  self.num_layers = getopt(opt, 'num_layers', 2)
  self.idx_to_token = getopt(opt, 'idx_to_token')
  self.dropout = getopt(opt, 'dropout', 0)
  opt.std = getopt(opt, 'std', 0.01)
  self.batch_size = getopt(opt, 'batch_size')
  self.beam_size = opt.beam_size
  opt.conv_feat_size = self.rnn_size
  opt.embedding_size = self.rnn_size
  self.max_width = getopt(opt, 'max_width')
  self.max_height = getopt(opt, 'max_height')
  self.output_width = self.max_width*self.max_height
  
  self.nets = {}
  self.nets.preprocess = nn.Sequential()
  self.nets.preprocess:add(nn.SpatialMaxPooling(1,6,1,1,0,0))
  self.nets.preprocess:add(nn.View(512, -1):setNumInputDims(3))
  self.nets.preprocess:add(nn.Transpose({2, 3}))  

  self.nets.enc_core = self:lstm(self.rnn_size, 1, self.rnn_size, self.num_layers, 0)

  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2
  self.sample_argmax = true
 
  local V, W = self.vocab_size, self.input_encoding_size
  self.nets.lookup_table = nn.LookupTable(V+2, W)

  self.nets.attention_nn = self:build_attention_nn(opt)
  self.lstm_input_size = self.input_encoding_size 
  self.nets.core = self:lstm(self.lstm_input_size, 1, self.rnn_size, self.num_layers, 0)
  
  self.nets.proj = nn.Sequential()
  self.nets.proj:add(nn.Linear(self.rnn_size*2, self.vocab_size+1))
  self.nets.proj:add(nn.LogSoftMax())

  self.core_output = torch.Tensor()
  self:_createInitState_enc(1)
  self:_createInitState(1)

  self:training()
end


function recog:build_attention_nn(opt)
  local conv_feat_maps = nn.Identity()()
  local prev_h = nn.Identity()()
  local mask = nn.Identity()()
  -- compute attention coefficients

  local orgfeat_embed = nn.View(self.output_width):setNumInputDims(3)(conv_feat_maps)
  local orgfeat_embed1 = nn.View(-1, opt.conv_feat_size, self.output_width)(orgfeat_embed)
  local orgfeat_embed2 = nn.Transpose({2, 3})(orgfeat_embed1)
  
  -- convolution operation
  local neighbor_feat = nn.SpatialConvolution(opt.conv_feat_size, opt.embedding_size, 3, 3, 1, 1, 1, 1)(conv_feat_maps)
  local X_embed = nn.View(self.output_width):setNumInputDims(3)(neighbor_feat)
  local X_embed1 = nn.View(-1, opt.embedding_size, self.output_width)(X_embed)
  local X_embed2 = nn.Transpose({2, 3})(X_embed1)
  -- Compute e
  -- Linear Transform: batch_size x input_size_g --> batch_size x embedding_size
  local g_embed = nn.Linear(opt.rnn_size, opt.embedding_size)(prev_h)
  -- Replicate: batch_size x embedding_size --> batch_size x seq_length_X x embedding_size
  local g_embed_replicate = nn.Replicate(self.output_width, 2)(g_embed)
  
   -- Add: batch_size x seq_length_X x embedding_size + batch_size x seq_length_X x embedding_size --> batch_size x seq_length_X x embedding_size
  local feat = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({X_embed2, g_embed_replicate})))
  -- Reshape: batch_size x seq_length_X x embedding_size --> (batch_size x seq_length_X) x embedding_size
  -- Linear Transform: (batch_size x seq_length_X) x embedding_size --> (batch_size x seq_length_X) x 1
  local e = nn.Linear(opt.embedding_size, 1)(nn.View(-1, opt.embedding_size)(feat))
  -- Compute attention weights
   -- Reshape: (batch_size x seq_length_X) x 1 --> batch_size x seq_length_X
   -- Softmax
   local alpha = nn.maskSoftMax()({nn.View(-1, self.output_width)(e),mask})
   -- Reshape: batch_size x seq_length_X --> batch_size x 1 x seq_length_X
   local alpha2 = nn.View(1,-1):setNumInputDims(1)(alpha)

   -- Compute attended feature
   -- Matrix Multiply: batch_size x <1 x seq_length_X，seq_length_X x input_size_X> = batch_size x 1 x input_size_X
   local Att = nn.MM(false, false)({alpha2, orgfeat_embed2})
   -- Reshape: batch_size x 1 x input_size_X --> batch_size x input_size_X
   local att_out = nn.View(-1, opt.conv_feat_size)(Att)

  -- -- create nn graph module
  return nn.gModule({conv_feat_maps, prev_h, mask}, {att_out, alpha})
end

function recog:lstm(input_size, output_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end


function recog:createClones()
  -- construct the net clones
  print('constructing clones inside the model')
  self.nets.image_encoder = {self.nets.enc_core}
  for it=2,self.max_width do
    self.nets.image_encoder[it] = self.nets.enc_core:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end

  self.nets.clones = {self.nets.core}
  self.nets.lookup_tables = {self.nets.lookup_table}
  self.nets.attention_nns = {self.nets.attention_nn}
  self.nets.projs = {self.nets.proj}

  for t=2,self.seq_length+2 do
    self.nets.clones[t] = self.nets.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.nets.lookup_tables[t] = self.nets.lookup_table:clone('weight', 'gradWeight')
    self.nets.attention_nns[t] = self.nets.attention_nn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.nets.projs[t] = self.nets.proj:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end


--decode sequence with caption scores
function recog:decodeSequence(seq,seqS)
  local delimiter = ' '
  local captions = {}
  local captions_Sco={}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    local caption_Sco = 0
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN or idx == 0 then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      if idx ~= 94 then
        caption = caption .. self.idx_to_token[idx]
        caption_Sco = caption_Sco + seqS[{i, t}]
      end
    end
    caption_Sco=caption_Sco/((string.len(caption)+1)/2)
    table.insert(captions, caption)
    table.insert(captions_Sco, caption_Sco)
  end
  return captions, captions_Sco
end

--simply decode the characters
function recog:decodeSequence_org(seq)
  local delimiter = ' '
  local captions = {}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    local caption_Sco = 0
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN or idx == 0 then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      if self.idx_to_token[idx] then
        caption = caption .. self.idx_to_token[idx]
      else
        caption = caption .. ''
      end
    end
    table.insert(captions, caption)
  end
  return captions
end

--create initial state for encoder LSTM
function recog:_createInitState_enc(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for encoder LSTM
  if not self.enc_init_state then self.enc_init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.enc_init_state[h] then
      if self.enc_init_state[h]:size(1) ~= batch_size then
        self.enc_init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.enc_init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.enc_num_state = #self.enc_init_state
end

--create initial state for decoder LSTM
function recog:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for decoder LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

--forward
function recog:updateOutput(input)
  self.recompute_backward = true
  local image_features = input[1]
  local gt_sequence = input[2]
  self.mask_ques = input[3]
  self.ques_seq_len = input[4]

  if self.nets.clones == nil then self:createClones() end
  
  if gt_sequence:nElement() > 0 then
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN

    self:_createInitState_enc(N)
    self:_createInitState(N)
    self.fore_state = {[0] = self.enc_init_state}
    self.fore_inputs = {}
    self.core_output:resize(N, self.max_width, self.rnn_size):zero()
    self.enc_tmax = self.max_width
    self.enc_tmin = torch.min(self.ques_seq_len)

    local image_vectors=self.nets.preprocess:forward(image_features)
    for t=1,self.enc_tmax do
        self.fore_inputs[t] = {image_vectors:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(self.fore_state[t-1])}
        local enc_out = self.nets.image_encoder[t]:forward(self.fore_inputs[t])
        if t > self.enc_tmin then
          for i=1,self.enc_num_state+1 do
            enc_out[i]:maskedFill(self.mask_ques:sub(1,-1,1,1,t,t):contiguous():view(N,1):expandAs(enc_out[i]):cudaByte(), 0)
          end
        end
        self.fore_state[t] = {} 
        for i=1,self.enc_num_state do table.insert(self.fore_state[t], enc_out[i]) end
        
        self.core_output:narrow(2,t,1):copy(enc_out[self.enc_num_state])
    end
    self.encode_output=torch.Tensor(N, self.rnn_size):zero():cuda()
    for ip =1, N do
      self.encode_output[ip]:copy(self.core_output[{{ip},{self.ques_seq_len[ip]},{}}]:squeeze())
    end

    self.state = {[0] = self.init_state}
    self.inputs = {}
    self.tmpcat = {}
    self.lookup_tables_inputs = {}
    self.tmax = 0 
    self.output=torch.Tensor(N, T+2, self.vocab_size+1):zero():cuda()
    self.mask=nn.View(-1,self.output_width):cuda():forward(self.mask_ques)
    local fordraw_attfeat=torch.Tensor(N, T+2, self.output_width):zero():cuda()

    for t=1,T+2 do
      local xt
      if t == 1 then
        xt = self.encode_output
      else
        local it = self._gt_with_start[{{},{t-1}}]:clone()
        it=it:view(-1)
        xt = self.nets.lookup_tables[t]:forward(it)
      end

      self.inputs[t] = {xt, unpack(self.state[t-1])}
      local out = self.nets.clones[t]:forward(self.inputs[t])
      self.state[t] = {}
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      
      local h_state = self.state[t][self.num_state]
      local att_all = self.nets.attention_nns[t]:forward({image_features, h_state, self.mask:contiguous()})
      local att = att_all[1]
      local attw =att_all[2]

      fordraw_attfeat:narrow(2,t,1):copy(attw)

      self.tmpcat[t] = torch.cat({h_state, att}, 2)
      local proj_out = self.nets.projs[t]:forward(self.tmpcat[t])
      self.output:narrow(2,t,1):copy(proj_out)
      self.tmax = t
    end
    self._forward_sampled = false
    return self.output
  else
    self._forward_sampled = true
    -- self.beam_size =5
    if self.beam_size ~= nil then
      print 'running beam search'
      self.output = self:beamsearch(image_features, self.beam_size)
      return self.output
    else
      return self:sample(image_features)
    end
  end
end



function recog:backward(input, gradOutput, scale)
  assert(self._forward_sampled == false, 'cannot backprop through sampling')
  assert(scale == nil or scale == 1.0)
  self.recompute_backward = false
  local image_features = input[1]
  local N = image_features:size(1)
  local dstate = {[self.tmax] = self.init_state}
  local dencodings = self.core_output:clone():zero()
  local dcontext = image_features:clone():zero()

  local gradOutput_rnn=nn.SplitTable(1, 2):cuda():forward(gradOutput)
  local dwt={}

  for t = self.tmax, 1, -1 do 
    local dproj = self.nets.projs[t]:backward(self.tmpcat[t], gradOutput_rnn[t])
    local dhstate = dproj[{{},{1, self.rnn_size}}]
    local dat = dproj[{{},{self.rnn_size+1, self.rnn_size*2}}]
    local dat_w = torch.Tensor(N, self.output_width):zero():cuda()

    -- backprop attention net
    local h_state = self.state[t][self.num_state]
    local datt = self.nets.attention_nns[t]:backward({image_features, h_state, self.mask:contiguous()}, {dat:contiguous(), dat_w:contiguous()})
    local dconv, dh_state, _ = unpack(datt)
    dhstate:add(dh_state)
    dcontext:add(dconv)

    local d_core_outt = torch.zeros(N, 1):cuda()
    local dout = {}
    for k=1,#dstate[t]-1 do table.insert(dout, dstate[t][k]) end
    table.insert(dout, dstate[t][self.num_state]:add(dhstate))
    table.insert(dout, d_core_outt)

    local dinputs = self.nets.clones[t]:backward(self.inputs[t], dout)
    local dxt = dinputs[1]:clone()
    dstate[t-1] = {} 
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
    
    if t == 1 then
      local dwt1 = torch.reshape(dxt, N, 1, self.rnn_size)
      for k = 1, N do
        dencodings[{{k},{self.ques_seq_len[k]},{}}]:add(dwt1[k])
      end
    else
      local it = self._gt_with_start[{{},{t-1}}]:clone()
      it=it:view(-1)
      self.nets.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
    end
  end
   -- go backwards and lets compute gradients
  local d_core_state = {[self.enc_tmax] = self.enc_init_state} 
  local d_core_outt = torch.zeros(N, 1):cuda()
  local d_embed_core = d_embed_core or self.core_output:new()
  d_embed_core:resize(N, self.max_width, self.rnn_size):zero()

  for t=self.enc_tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    table.insert(dout, d_core_state[t][1]) 
    table.insert(dout, d_core_state[t][2]) 
    table.insert(dout, d_core_state[t][3]) 
    table.insert(dout, dencodings:narrow(2,t,1):contiguous():view(-1, self.rnn_size))
    table.insert(dout, d_core_outt) 
    local dinputs = self.nets.image_encoder[t]:backward(self.fore_inputs[t], dout)
    if t > self.enc_tmin then
      for k=1,self.enc_num_state+1 do
        dinputs[k]:maskedFill(self.mask_ques:sub(1,-1,1,1,t,t):contiguous():view(N,1):expandAs(dinputs[k]):cudaByte(), 0)
      end
    end
    d_core_state[t-1] = {} -- copy over rest to state grad
    for k=2,self.enc_num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
    d_embed_core:narrow(2,t,1):copy(dinputs[1])
  end

  local gradInput_t11 = self.nets.preprocess:backward(image_features, d_embed_core)
  local gradInput_t1 = torch.add(gradInput_t11,dcontext)
  local gradInput_t2 = input[2].new(#input[2]):zero()
  local gradInput_t3 = input[3].new(#input[3]):zero()
  local gradInput_t4 = input[4].new(#input[4]):zero()
  
  self.gradInput={gradInput_t1, gradInput_t2, gradInput_t3, gradInput_t4}
  return self.gradInput
end

--[[
Convert a ground-truth sequence of shape to a target suitable for the
TemporalCrossEntropyCriterion.

Input:
- gt_sequence: Tensor of shape (N, T) where each element is in the range [0, V];
  an entry of 0 is a null token.
--]]
function recog:getTarget(gt_sequence)

  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 2):zero()
  target[{{}, {2, T + 1}}]:copy(gt_sequence)
  for i = 1, N do
    for t = 2, T + 2 do
      if target[{i, t}] == 0 then
        -- Replace the first null with an end token
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  return target:type(gt_sequence:type())
end

-- Greedy decoding by taking the symbol with the highest softmax score
function recog:sample(image_features)
  local N, T = image_features:size(1), self.seq_length
  local seq = torch.LongTensor(N, T+2):zero()
  local seqScore = torch.Tensor(N, T+2):zero()
  local scores, scores_t
  local fordraw_attfeat=torch.Tensor(N, T+2, self.output_width):zero():cuda()
  local pre_output=torch.Tensor(N, T+2, self.vocab_size+1):zero():cuda()

  self:_createInitState_enc(N)
  local fore_state = {[0] = self.enc_init_state}
  local fore_inputs = {}
  local core_output=torch.Tensor(N, self.max_width, self.rnn_size):zero():cuda()
  local enc_tmax = self.max_width
  local enc_tmin = torch.min(self.ques_seq_len)

  local image_vectors=self.nets.preprocess:forward(image_features)
  for t=1,enc_tmax do
      fore_inputs[t] = {image_vectors:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(fore_state[t-1])}
      local enc_out = self.nets.enc_core:forward(fore_inputs[t])
      if t > enc_tmin then
        for i=1,self.enc_num_state+1 do
          enc_out[i]:maskedFill(self.mask_ques:sub(1,-1,1,1,t,t):contiguous():view(N,1):expandAs(enc_out[i]):cudaByte(), 0)
        end
      end
      fore_state[t] = {} -- the rest is state
      for i=1,self.enc_num_state do table.insert(fore_state[t], enc_out[i]) end
      core_output:narrow(2,t,1):copy(enc_out[self.enc_num_state])
  end
  local temp_encode_output=torch.Tensor(N, self.rnn_size):zero():cuda()

  for ip =1, N do
    temp_encode_output[ip]:copy(core_output[{{ip},{self.ques_seq_len[ip]},{}}]:squeeze(1))
  end

  self:_createInitState(N)
  local state = self.init_state
  local mask=nn.View(-1,self.output_width):cuda():forward(self.mask_ques)

  for t=1,T+2 do
    local xt, it
    if t == 1 then
      xt = temp_encode_output
    elseif t ==2 then
      it = torch.LongTensor(N):fill(self.START_TOKEN)
      xt = self.nets.lookup_table:forward(it)
    else
      it = seq[{{}, {t-1, t-1}}]:clone()
      it = it:view(-1):long()
      xt = self.nets.lookup_table:forward(it)     
    end

    local inputs = {xt,unpack(state)}
    local out = self.nets.core:forward(inputs)
    state = {}
    for j=1,self.num_state do table.insert(state, out[j]) end
    
    -- get attention feature
    local h_state = state[self.num_state]
    local att_all = self.nets.attention_nn:forward({image_features, h_state, mask:contiguous()})
    local att = att_all[1]
    local attw =att_all[2]

    fordraw_attfeat:narrow(2,t,1):copy(attw)
    local tmpcat = torch.cat(h_state, att)
    local scores = self.nets.proj:forward(tmpcat)
    pre_output:narrow(2,t,1):copy(scores)
  
    local idxscore, idx = torch.max(scores, 2)
    local idxscore2 = torch.exp(idxscore)
    seq[{{}, t}]:copy(idx)
    seqScore[{{}, t}]:copy(idxscore2)
  end
  self.output = {pre_output, seq[{{},{2,T+2}}], seqScore[{{},{2,T+2}}], fordraw_attfeat}

  return self.output
end


--using beamsearch for decoding by maintaining the 'beam_size' candidates
function recog:beamsearch(image_features, beam_size)
  beam_size = beam_size or 20
  local N, T = image_features:size(1), self.seq_length
  local seq = torch.LongTensor(N, T+1):zero():cuda()
  local seqScore = torch.Tensor(N, T+1):zero():cuda()
  local scores, scores_t
  local fordraw_attfeat=torch.Tensor(N, T+2, self.output_width):zero():cuda()
  local pre_output=torch.Tensor(N, T+2, self.vocab_size+1):zero():cuda()

  self:_createInitState_enc(N)
  local fore_state = {[0] = self.enc_init_state}
  local fore_inputs = {}
  local core_output=torch.Tensor(N, self.max_width, self.rnn_size):zero():cuda()
  local enc_tmax = self.max_width
  local enc_tmin = torch.min(self.ques_seq_len)

  local image_vectors=self.nets.preprocess:forward(image_features)
  for t=1,enc_tmax do
      fore_inputs[t] = {image_vectors:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(fore_state[t-1])}
      local enc_out = self.nets.enc_core:forward(fore_inputs[t])
      if t > enc_tmin then
        for i=1,self.enc_num_state+1 do
          enc_out[i]:maskedFill(self.mask_ques:sub(1,-1,1,1,t,t):contiguous():view(N,1):expandAs(enc_out[i]):cudaByte(), 0)
        end
      end
      fore_state[t] = {} -- the rest is state
      for i=1,self.enc_num_state do table.insert(fore_state[t], enc_out[i]) end
      core_output:narrow(2,t,1):copy(enc_out[self.enc_num_state])
  end
  local temp_encode_output=torch.Tensor(N, self.rnn_size):zero():cuda()

  for ip =1, N do
    temp_encode_output[ip]:copy(core_output[{{ip},{self.ques_seq_len[ip]},{}}]:squeeze(1))
  end

  local mask=nn.View(-1,self.output_width):cuda():forward(self.mask_ques)
  for i = 1, N do
    self:_createInitState(1)
    local state = self.init_state
    local image_features_i=image_features[{{i},{},{},{}}]
    local mask_i=mask[{{i},{}}]:contiguous()

    local xt = temp_encode_output[{{i},{}}]
    local inputs = {xt,unpack(state)}
    local out = self.nets.core:forward(inputs)
    state = {}
    for j=1,self.num_state do table.insert(state, out[j]) end
    
    -- get attention feature
    local h_state = state[self.num_state]
    local att_all = self.nets.attention_nn:forward({image_features_i, h_state, mask_i})
    local att = att_all[1]
    local attw =att_all[2]

    local tmpcat = torch.cat(h_state, att)
    local scores = self.nets.proj:forward(tmpcat)

    local it = torch.LongTensor(1):fill(self.START_TOKEN)
    xt = self.nets.lookup_table:forward(it)
    local inputs = {xt,unpack(state)}
    local out = self.nets.core:forward(inputs)
    state = {}
    for j=1,self.num_state do table.insert(state, out[j]) end
    -- get attention feature
    local h_state = state[self.num_state]
    local att_all = self.nets.attention_nn:forward({image_features_i, h_state, mask_i})
    local att = att_all[1]
    local attw =att_all[2]

    local tmpcat = torch.cat(h_state, att)
    local scores = self.nets.proj:forward(tmpcat)

    -- Initialize our beams to the words with the highest logprobs
    local beams = seq.new(beam_size, T+1):fill(self.NULL_TOKEN):cuda()
    local beams_pro = seq.new(beam_size, T+1):fill(1.0):cuda()
    
    local beam_logprobs, idx = torch.topk(scores, beam_size, 2, true)

    idx=idx:squeeze()
    beams[{{}, 1}]:copy(idx)
    beams_pro[{{}, 1}]:copy(beam_logprobs)

    for il=1,self.num_state do 
      state[il]=state[il]:expand(beam_size, self.rnn_size):clone()
    end

    image_features_i=image_features_i:expand(beam_size, image_features_i:size(2), image_features_i:size(3), image_features_i:size(4)):clone()
    mask_i = mask_i:expand(beam_size, mask_i:size(2)):clone()

    for t=2, T+1 do
      it = beams[{{}, {t - 1, t - 1}}]:long()
      it = it:view(-1):long()
      xt = self.nets.lookup_table:forward(it)     

      local inputs = {xt,unpack(state)}
      local out = self.nets.core:forward(inputs)
      state = {}
      for j=1,self.num_state do table.insert(state, out[j]) end

      -- get attention feature
      local h_state = state[self.num_state]
      local att_all = self.nets.attention_nn:forward({image_features_i, h_state, mask_i:contiguous()})
      local att = att_all[1]
      local attw =att_all[2]
     
      local tmpcat = torch.cat(h_state, att)
      local next_word_logprobs = self.nets.proj:forward(tmpcat)

      -- If a beam already has an END token then any subsequent words should
      -- not contribute to its logprobs, so set them to zero
      local end_mask = torch.eq(torch.eq(beams, self.END_TOKEN):sum(2), 0)
      end_mask = end_mask:type(next_word_logprobs:type())
      next_word_logprobs:cmul(end_mask:expandAs(next_word_logprobs))

      -- For each beam, find the top beam_size next words
      local top_next_word_logprobs, word_idx
        = torch.topk(next_word_logprobs, beam_size, 2, true)

      local beam_logprobs_dup = beam_logprobs:view(-1, 1)
                                  :expand(beam_size, beam_size)
                                  :contiguous()
                                  :view(beam_size * beam_size)

      local all_next_logprobs = top_next_word_logprobs:view(-1)
                                   + beam_logprobs_dup

      beam_logprobs, idx = torch.topk(all_next_logprobs, beam_size, 1, true)
      local all_next_beams = beams:view(beam_size, 1, T+1)
                               :expand(beam_size, beam_size, T+1)
                               :contiguous()
                               :view(beam_size * beam_size, T+1)

      local all_next_beams_pro = beams_pro:view(beam_size, 1, T+1)
                               :expand(beam_size, beam_size, T+1)
                               :contiguous()
                               :view(beam_size * beam_size, T+1)

      all_next_beams[{{}, t}]:copy(word_idx:view(-1))
      all_next_beams_pro[{{}, t}]:copy(top_next_word_logprobs:view(-1))

      beams = all_next_beams:index(1, idx)
      beams_pro = all_next_beams_pro:index(1, idx)

      for il=1,self.num_state do 
        local H = state[il]:size(2)
        local state_dup = state[il]:view(beam_size, 1, H)
                                   :expand(beam_size, beam_size, H)
                                   :contiguous()
                                   :view(beam_size * beam_size, H)
        state[il] = state_dup:index(1, idx)
      end
    end

    -- After running over all timesteps, copy best beam to seq
    local _, best_beam_idx = beam_logprobs:max(1)
    seq[i] = beams[best_beam_idx[1]]
    seqScore[i] = beams_pro[best_beam_idx[1]]
  end

  seqScore = torch.exp(seqScore)
  local empty = seq.new()
  self.output = {empty, seq, seqScore, empty}
  return self.output
end

function recog:training()
  parent.training(self)
  if self.nets.clones == nil then self:createClones() end 
  self.nets.preprocess:training()
  for k,v in pairs(self.nets.image_encoder) do v:training() end
  for k,v in pairs(self.nets.clones) do v:training() end
  for k,v in pairs(self.nets.lookup_tables) do v:training() end
  for k,v in pairs(self.nets.attention_nns) do v:training() end
  for k,v in pairs(self.nets.projs) do v:training() end
end


function recog:evaluate()
  parent.evaluate(self)
  if self.nets.clones == nil then self:createClones() end
  self.nets.preprocess:evaluate()
  self.nets.enc_core:evaluate()
  self.nets.core:evaluate()
  self.nets.lookup_table:evaluate()
  self.nets.attention_nn:evaluate()
  self.nets.proj:evaluate()
  for k,v in pairs(self.nets.image_encoder) do v:evaluate() end
  for k,v in pairs(self.nets.clones) do v:evaluate() end
  for k,v in pairs(self.nets.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.nets.attention_nns) do v:evaluate() end
  for k,v in pairs(self.nets.projs) do v:evaluate() end
end


function recog:clearState()
  if self.nets.clones == nil then self:createClones() end
  self.nets.preprocess:clearState()
  for k,v in pairs(self.nets.image_encoder) do v:clearState() end
  for k,v in pairs(self.nets.clones) do v:clearState() end
  for k,v in pairs(self.nets.lookup_tables) do v:clearState() end
  for k,v in pairs(self.nets.attention_nns) do v:clearState() end
  for k,v in pairs(self.nets.projs) do v:clearState() end
end


function recog:getModulesList()
  return {self.nets.enc_core, self.nets.lookup_table, self.nets.core, self.nets.attention_nn, self.nets.proj}
end

function recog:parameters()
  -- flatten model parameters and gradients into single vectors
  local params, grad_params = {}, {}
  for k, m in pairs(self:getModulesList()) do
    local p, g = m:parameters()
    for _, v in pairs(p) do table.insert(params, v) end
    for _, v in pairs(g) do table.insert(grad_params, v) end
  end
  -- invalidate clones as weight sharing breaks
  -- self.nets.image_encoder = nil
  self.nets.clones = nil
  self.nets.lookup_tables = nil
  self.nets.attention_nns = nil
  self.nets.projs = nil
 
  -- return all parameters and gradients
  return params, grad_params
end
