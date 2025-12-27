import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';

class BaseModel{

  constructor() {
    this.capturedDataset = {};
    this.trainXs = [];
    this.trainYs = [];
    this.embeddingSize = 1000;
    this.model = null; // The classification head "model" that takes the outputed embeddings from mobilenet
    this.trainingStatus = 0; // 0: not trained, 1: training, 2: training stopped, 3: trained , 4: imported
    this.NUM_CLASSES = 3;
    this.IMAGE_SIZE = 227;

    // Create the whole model
    this.createBackboneModel()
    this.createClassificationHead()

  }

  
  // You could comment it, I just added it to give you an idea
  addToDict(key, value) {

    // key = class index : 0, 1, 2, ...
    // value = base64 image data URL
    if (!this.capturedDataset[key]) {
      this.capturedDataset[key] = [];   // create list if not exists
    }
    this.capturedDataset[key].push(value);
  }


  async createBackboneModel(){
    this.mobilenet = await mobilenetModule.load();
  }


  async createClassificationHead() {

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [this.embeddingSize],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.dense({
      units: this.NUM_CLASSES,
      activation: 'softmax',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }


  async imageBase64ToTensor(base64DataUrl) {

    // 1. Create a new Image object
    const img = new Image();
    img.crossOrigin = "Anonymous"; // Handle potential CORS issues
    
    // 2. Wrap the onload in a Promise to handle asynchronous loading
    const imgLoadPromise = new Promise((resolve, reject) => {
        img.onload = () => resolve(img);
        img.onerror = reject;
    });

    // 3. Set the source to the Data URL
    img.src = base64DataUrl;

    // 4. Wait for the image to load
    const loadedImg = await imgLoadPromise;

    // 5. Convert the loaded image element into a tensor
    const tensor = tf.fromPixels(loadedImg);
        
    return tensor;
  }


  async convertUrlToEmbedding(trainData){

      // The outputed logits from mobilenet
      let logits;
      this.createClassificationHead();
      
      for (const key in trainData) {

        for(const imageURL of trainData[key]) {

          // Convert base64 image to tensor          
          const image = await this.imageBase64ToTensor(imageURL);
          
          // 'conv_preds' is the logits activation of MobileNet.
          logits = this.mobilenet.infer(image, 'conv_preds');

          // Convert logits to 2D embedding
          const emb = logits.as2D(1, -1);

          // Store the embedding and the label
          this.trainXs.push(emb.clone());
          this.trainYs.push(tf.oneHot(tf.tensor1d([key]).toInt(), this.NUM_CLASSES));

          // Dispose tensors to free memory
          image.dispose();
          logits.dispose();
          emb.dispose();

        }
      }


  }


  async trainModel(trainData, epochs, batchSize_, lr) {

    console.log("Preparing data for training...");

    // Convert all images to embeddings
    await this.convertUrlToEmbedding(trainData);

    if (this.trainXs.length === 0) {
      console.log("No examples to train on");
      return;
    }

    await tf.nextFrame();

    // Stack examples
    const xs = tf.concat(this.trainXs, 0);
    const ys = tf.concat(this.trainYs, 0);

    console.log("Training...");
    
    // Define the batch size
    const batchSize = Math.min(batchSize_, xs.shape[0]);

    // Define the learning rate
    const LEARNING_RATE = lr; // Common small positive value

    // Create an optimizer with the specified learning rate
    // For example, using the Adam optimizer, which is a popular choice
    const optimizer = tf.train.adam(LEARNING_RATE); 

    // Compile the model, specifying the optimizer, loss function, and metrics
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy', // Example loss function
      metrics: ['accuracy'], // Example metric
    });


    // Train the model
    await this.model.fit(xs, ys, {
      batchSize,
      epochs,
      shuffle: true,
      // learningRate: lr,
      callbacks: {
        onTrainBegin: async () => {
          this.trainingStatus = 1; // set status to training
          console.log("Training started");
        },

        onEpochEnd: async (epoch, logs) => {
          this.reportProgress(epoch, epochs, logs);
          await tf.nextFrame();
        },

        onBatchEnd: async (batch, logs) => {
          if (this.trainingStatus === 2) {
            this.model.stopTraining = true
          }
          await tf.nextFrame();
        },

        onTrainEnd: async () => {
          this.trainingStatus = 3; // set status to trained
          this.reportTrainingDone();
        }
      }
    });

    xs.dispose();
    ys.dispose();

    // this.modelTrained = !this.stopTrainingFlag ? true : false;
    // this.modelIsImported = false;
    console.log(this.trainingStatus === 3 ? 'Training completed' : 'Training stopped');
  }


  reportProgress(epoch, epochs, logs)
  {
    console.log(`Epoch ${epoch + 1} / ${epochs}: loss = ${logs.loss.toFixed(3)}, accuracy = ${logs.acc !== undefined ? logs.acc.toFixed(3) : (logs.accuracy || 0).toFixed(3)}`);

  }


  stopTraining() {
    if (this.trainingStatus === 1){
      this.trainingStatus = 2; // set status to training stopped
      return true;
    }
    return false;
  }


  reportTrainingDone()
  {
    console.log("TrainingDone");
  }


  async test(source){

    // Source could be image, video or canvas element
    const image = tf.fromPixels(source);

    // Define the logits
    let logits;

    // 'conv_preds' is the logits activation of MobileNet.
    const infer = () => this.mobilenet.infer(image, 'conv_preds');

    // Infering
    logits = infer();

    // Convert is as 2D array to feed it to the classification model
    const emb = logits.as2D(1, -1);
    const preds = this.model.predict(emb);
    
    const probs = await preds.data(); // probs is Float32Array of the prediction probabilities of each class.
    const classIndex = probs.indexOf(Math.max(...probs)); // classIndex is the index of the highest predicted class.
    
    return {probs, classIndex};

  }


  async buildConfustionMatrix() {

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
      this.NUM_CLASSES
    );

    // Print the resulting Confusion Matrix Tensor
    confusionMatrix.print();
  }


}


class Main extends BaseModel {

  constructor() {
    super();
    // Initiate variables
    this.training = -1; // -1 when no class is being captured
    this.videoPlaying = false;
    this.canvas = document.getElementById("canvas");
    this.ctx = canvas.getContext("2d");
        

    // Get the video element
    this.video = document.getElementsByTagName('video')[0];

    // Create training buttons and info texts    
    for (let i = 0; i < this.NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';
      div.style.marginTop = '16px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Capture class " + i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      // this.infoTexts.push(infoText);
    }


    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = this.IMAGE_SIZE;
        this.video.height = this.IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })

      this.bindPage();
  }


  async bindPage() {
    // this.mobilenet = await mobilenetModule.load();
    this.start();
  }


  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }


  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }


  async animate() {

    if (this.videoPlaying) {

      // Capture examples if one of the buttons is held down
      if (this.training != -1) {

        // Draw the video frame to the canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Convert to image data or base64 image
        const dataURL = this.canvas.toDataURL("image/png");
        
        // Add the image to the dataset object
        this.addToDict(this.training, dataURL);

      }


      if (this.trainingStatus == 3 || this.trainingStatus == 4) {

        // If the model is trained run test
        const {probs, classIndex} = await this.test(this.video);
        console.log(classIndex);

      }

    
    }

    this.timer = requestAnimationFrame(this.animate.bind(this));
  }


}


window.addEventListener('load', () => {
  window.app = new Main();
});
