from utils import importDataInfo, balanceData, loadData, createModel, batchGen
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import matplotlib as plt
#step 1
path = 'myData'
data = importDataInfo(path)

# step 2
data = balanceData(data,display=False)

# Step 3: Prepare for processing
imagesPath, steerings = loadData(path,data)

# Step 4: Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))


#### STEP 5: Augmentation

#### STEP 8
model = createModel()
model.summary()

#### STEP  9
history = model.fit(
    batchGen(xTrain, yTrain, 10, 1),
    steps_per_epoch=20,
    epochs=2,
    validation_data=batchGen(xVal, yVal, 10, 0),
    validation_steps=20
)

#### STEP 10
model.save('model.h5')
print('Model Saved')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend (['Training', 'Validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.show()
