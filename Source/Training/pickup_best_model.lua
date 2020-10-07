--- Pick up the model with lowest validation loss

require 'torch'

require 'nn'
require 'cunn'

local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'

if #arg == 0 then
  print("Please specify the street. 1 = preflop, 4 = river")
  return
end

local street = tonumber(arg[1])


function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

function select_best_model(street)

  print('Selecting best model with less Validation Huber Loss ...')

  local best_loss = 1
  local best_epoch = 0
  local best_model_info_path = ''

  local epoch_count = arguments.epoch_count
  local save_epoch = arguments.save_epoch

  local path = arguments.model_path
  if game_settings.nl then
    path = path .. "NoLimit/"
  else
    path = path .. "Limit/"
  end

  if street == 4 then
    path = path .. "river/"
  elseif street == 3 then
    path = path .. "turn/"
  elseif street == 2 then
    path = path .. "flop/"
  elseif street == 1 then
    path = path .. "preflop-aux/"
  end

  local net_type_str = arguments.gpu and '_gpu' or '_cpu'

  for epoch = 1, epoch_count do

    if (epoch % save_epoch == 0) then

      local information_file_name = path .. 'epoch_' .. epoch .. net_type_str .. '.info'

      if file_exists(information_file_name) then
        table = torch.load(information_file_name)
        if (table.valid_loss < best_loss) then
          best_loss = table.valid_loss
          best_epoch = epoch
          best_model_info_path = information_file_name
        end
      end
    end
  end

  if file_exists(best_model_info_path) then

    print('best epoch: ' .. best_epoch)
    print('best loss: ' .. best_loss)
    print('best model info path ' .. best_model_info_path)

    print('saving final model')

    local best_model_path = path .. 'epoch_' .. best_epoch .. net_type_str .. '.model'

    if file_exists(best_model_path) then
      local best_model = torch.load(best_model_path)
      local final_model_file_name = path .. 'final_' .. net_type_str .. '.model'

      torch.save(final_model_file_name, best_model)

    else
      print('error finding best model to pickup -- not found ' .. best_model_path)
    end


    local best_model_information = torch.load(best_model_info_path)
    local final_information_file_name = path .. 'final_' .. net_type_str .. '.info'

    torch.save(final_information_file_name, best_model_information)

  else

    print('error finding best model to pickup -- there are no models')
  end

end

select_best_model(street)
