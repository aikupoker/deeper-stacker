--- Pick up the model with lowest validation loss
require 'torch'

best_loss = 1
best_epoch = 0
-- Change the iter number to fit your models
for i=1,300 do
  table = torch.load('epoch_' .. i .. '_gpu.info')
  if (table.valid_loss < best_loss) then
    best_loss = table.valid_loss
    best_epoch = i
  end
end

print('best epoch: ' .. best_epoch)
print('best loss: ' .. best_loss)
