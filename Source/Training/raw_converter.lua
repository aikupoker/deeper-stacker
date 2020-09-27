if #arg == 0 then
  print("Please specify the street. 1 = preflop, 4 = river")
  return
end

require 'torch'
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local bucketer = require 'Nn.bucketer'
local river_tools = require 'Nn.Bucketing.river_tools'
local constants = require 'Settings.constants'
local card_to_string = require 'Game.card_to_string_conversion'
local card_tools = require 'Game.card_tools'
local evaluator = require 'Game.Evaluation.evaluator'
local threads = require 'threads'

local DataStreamMem = torch.class('RawConverter')

require 'Nn.bucket_conversion'

-- Lua implementation of PHP scandir function
function scandir(directory)
  local i, t, popen = 0, {}, io.popen
  local pfile = popen('ls -a "' .. directory .. '"')
  for filename in pfile:lines() do
    i = i + 1
    t[filename] = 1
  end
  pfile:close()
  return t
end

function convert(street)

  local srcfolder = "srcfolder"
  local destfolder = "destfolder"

  if street == 4 then
    srcfolder = "river_raw/"
    destfolder = "river/"
  elseif street == 3 then
    srcfolder = "turn_raw/"
    destfolder = "turn/"
  elseif street == 2 then
    srcfolder = "flop_raw/"
    destfolder = "flop/"
  elseif street == 1 then
    srcfolder = "preflop-aux_raw/"
    destfolder = "preflop-aux/"
  end

  --loading valid data
  local path = arguments.data_path
  if game_settings.nl then
    path = path .. "NoLimit/"
  else
    path = path .. "Limit/"
  end
  filenames = scandir(path .. srcfolder)
  local numfiles = 0
  local goodfiles = {}
  for filename,_ in pairs(filenames) do
    local res = string.find(filename, ".inputs")
    if res ~= nil then
      local targetname = filename:sub(0, res) .. "targets"
      if filenames[targetname] ~= nil then
        goodfiles[filename:sub(0,res)] = 1
        numfiles = numfiles + 1
      end
    end
  end

  local bucket_conversion = BucketConversion()

  print(numfiles .. " good files")
  local bucket_count = bucketer:get_bucket_count(street)
  local target_size = bucket_count * constants.players_count

  -- ranges, termvalues, potsize
  local input_size = bucket_count * constants.players_count + 1

  local fileidx = 1
  local file_pattern = "[-](......"
  if street == 1 then
    file_pattern = ""
  elseif street == 2 then
    file_pattern = file_pattern .. ")"
  elseif street == 3 then
    file_pattern = file_pattern .. "..)"
  elseif street == 4 then
    file_pattern = file_pattern .. "....)"
  end

  local input_batch = arguments.Tensor(arguments.gen_batch_size, input_size)
  local target_batch = arguments.Tensor(arguments.gen_batch_size, target_size)

  for filebase, _ in pairs(goodfiles) do
    local inputname = filebase .. "inputs"
    local targetname = filebase .. "targets"

    if street > 1 then
      local board = card_to_string:string_to_board(string.match(filebase, file_pattern))
      bucket_conversion:set_board(board)
    else
      bucket_conversion:set_board(arguments.Tensor())
    end

    local raw_input_batch = torch.load(path .. srcfolder .. inputname)
    local raw_target_batch = torch.load(path .. srcfolder .. targetname)

    --input_batch

    fileidx = fileidx + 1
    if fileidx % 100 == 0 then
      print(fileidx .. "/" .. numfiles)
    end

    local raw_indexes = {{1, game_settings.hand_count}, {game_settings.hand_count + 1, game_settings.hand_count * 2}}
    local bucket_indexes = {{1, bucket_count}, {bucket_count + 1, bucket_count * 2}}

    for player = 1, constants.players_count do
      local player_index = raw_indexes[player]
      local bucket_index = bucket_indexes[player]
      bucket_conversion:card_range_to_bucket_range(raw_input_batch[{{},player_index}],input_batch[{{}, bucket_index}])
    end

    for player = 1, constants.players_count do
      local player_index = raw_indexes[player]
      local bucket_index = bucket_indexes[player]
      bucket_conversion:hand_cfvs_to_bucket_cfvs(
        raw_input_batch[{{}, player_index}],
        raw_target_batch[{{}, player_index}],
        input_batch[{{}, bucket_index}],
        target_batch[{{}, bucket_index}])
    end
    input_batch[{{}, -1}]:copy(raw_input_batch[{{}, -1}])

    if arguments.gpu then
      torch.save(path .. destfolder .. targetname, target_batch:cuda())
      torch.save(path .. destfolder .. inputname, input_batch:cuda())
    else
      torch.save(path .. destfolder .. targetname, target_batch:float())
      torch.save(path .. destfolder .. inputname, input_batch:float())
    end

  end
end

convert(tonumber(arg[1]))
