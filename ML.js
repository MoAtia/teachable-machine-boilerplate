import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';

function reportProgress(epoch, epochs, logs)
{
  console.log(`Epoch ${epoch + 1} / ${epochs}: loss = ${logs.loss.toFixed(3)}, accuracy = ${logs.acc !== undefined ? logs.acc.toFixed(3) : (logs.accuracy || 0).toFixed(3)}`);

}

function TrainingDone()
{
  console.log("TrainingDone")
}

// Number of classes to classify
const NUM_CLASSES = 4;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;

class ML {
  constructor() {
    this.trainingStatus = 0; // 0: Idle, 1: Training, 2: Stopped, 3: Done
    this.mobilenet = null;
    this.model = null;
    this.trainXs = [];
    this.trainYs = [];
  }

  async createBackboneModel() {
    this.mobilenet = await mobilenetModule.load();
    console.log("MobileNet Loaded");
  }

  createClassificationHead(inputShape) {
    if (this.model) return;
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({inputShape: [inputShape], units: 100, activation: 'relu'}));
    this.model.add(tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'}));
  }

  async convertUrlToEmbedding(trainingData) {
    const classes = Object.keys(trainingData);
    for (const label of classes) {
      const classId = parseInt(label.replace('class', ''));
      for (const url of trainingData[label]) {
        const img = await this.loadImage(url);
        const embedding = tf.tidy(() => {
           const image = tf.fromPixels(img);
           const logits = this.mobilenet.infer(image, 'conv_preds');
           return logits.as2D(1, -1);
        });
        this.trainXs.push(embedding);
        this.trainYs.push(tf.tidy(() => tf.oneHot(tf.tensor1d([classId], 'int32'), NUM_CLASSES)));
      }
    }
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = src;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => resolve(img);
      img.onerror = reject;
    });
  }

  // --- Abstracted Interfaces ---

  async startTraining(trainingData, epochs, batchSize, lr) {
    // TODO add ReportProgress() function
    // TODO add TrainingDone() function
    this.trainXs = [];
    this.trainYs = [];

    // Convert the URL object into embeddings
    await this.convertUrlToEmbedding(trainingData);

    if (this.trainXs.length === 0) return console.error("No data!");

    this.createClassificationHead(this.trainXs[0].shape[1]);

    const xs = tf.concat(this.trainXs, 0);
    const ys = tf.concat(this.trainYs, 0);

    const optimizer = tf.train.adam(lr);
    this.model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    await this.model.fit(xs, ys, {
      batchSize: Math.min(batchSize, xs.shape[0]),
      epochs: epochs,
      callbacks: {
        onTrainBegin: () => { this.trainingStatus = 1; },
        onEpochEnd: async (epoch, logs) => {
          reportProgress(epoch, epochs, logs);
          // await tf.nextFrame();
        },
        onTrainEnd: () => { 
          this.trainingStatus = 3;
          TrainingDone();
        }
      }
    });

    xs.dispose();
    ys.dispose();
  }


  stopTraining() {
    if (this.trainingStatus !== 1) return false;
    if (this.model) this.model.stopTraining = true;
    this.trainingStatus = 2;
    return true;
  }

  displayConfusionMatrix()
  {
    
    // Stack examples
    const xs = tf.concat(this.trainXs, 0);
    const ys = tf.concat(this.trainYs, 0);

    // Get the predictions from the model
    const rawPredictions = this.model.predict(xs);

    rawPredictions.print();

    const predictedClassIndices = tf.argMax(rawPredictions, 1);
    console.log(predictedClassIndices)

    const trueClassIndices = tf.argMax(ys, 1);
    console.log(trueClassIndices)
    // trueClassIndices.max().print();

    const maxTrueClassIndex = tf.max(trueClassIndices).dataSync()[0];
    console.log("Max true class index:", maxTrueClassIndex);
    
    const confusionMatrix = tf.math.confusionMatrix(
      trueClassIndices,
      predictedClassIndices,
      4
    );

    // Print the resulting Confusion Matrix Tensor
    confusionMatrix.print();
    return confusionMatrix.dataSync();
  }

  async exportModel() {
    await this.model.save('downloads://my-model');
    return true;
  }

  async importModel(jsonFile, weightsFiles) {
    this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...weightsFiles]));
    this.trainingStatus = 3;
    return true;
  }

  async test(source) { // TODO change the logic to accept image URL as input
    if (!this.model) return null;
    
    return tf.tidy(() => {
        const image = tf.fromPixels(source);
        const logits = this.mobilenet.infer(image, 'conv_preds');
        const emb = logits.as2D(1, -1);
        const preds = this.model.predict(emb);
        
        // Convert tensor to array
        const probs = preds.dataSync(); 
        const classIndex = preds.argMax(1).dataSync()[0];
        
        return {
            probs: Array.from(probs), 
            classIndex: classIndex 
        };
    });
  }
}

window.ML = ML;
