// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';

// Number of classes to classify
const NUM_CLASSES = 2;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;


class Main {

  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being captured
    this.videoPlaying = false;
    this.exampleCounts = new Array(NUM_CLASSES).fill(0);
    this.trainXs = [];
    this.trainYs = [];
    this.model = null;
    this.modelTrained = false;
    this.embeddingSize = 1000;
    this.capturedDataset = {};
    this.trainDataset = {};
    this.model = null;
    this.modelIsImported = false;

    this.canvas = document.getElementById("canvas");
    this.ctx = canvas.getContext("2d");


    // Initiate the page (load mobilenet, etc.)
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');
    this.video.style.transform = 'scaleX(-1)'; // Flip the video horizontally

    // Add video element to DOM
    document.body.appendChild(this.video);

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

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
      this.infoTexts.push(infoText);
    }

    // Global training button
    const trainDiv = document.createElement('div');
    trainDiv.style.marginTop = '16px';
    const trainBtn = document.createElement('button');
    trainBtn.innerText = 'Train Model';
    trainDiv.appendChild(trainBtn);
    this.trainStatus = document.createElement('span');
    this.trainStatus.style.marginLeft = '8px';
    this.trainStatus.innerText = ' Idle';
    trainDiv.appendChild(this.trainStatus);
    document.body.appendChild(trainDiv);

    // Export model button
    const saveModeldiv = document.createElement('div');
    saveModeldiv.style.marginTop = '16px';
    const saveModelBtn = document.createElement('button');
    saveModelBtn.innerText = 'Save Model';
    saveModeldiv.appendChild(saveModelBtn);
    document.body.appendChild(saveModeldiv);

    
    // import model buttons
    // 2. Create the Model File Input

    const loadModeldiv = document.createElement('div');
    loadModeldiv.style.marginTop = '16px';

    const modelInput = document.createElement('input');
    modelInput.type = 'file';
    modelInput.id = 'model-file-input';
    modelInput.accept = '.json';

    // 3. Create the Weights File Input
    const weightsInput = document.createElement('input');
    weightsInput.type = 'file';
    weightsInput.id = 'weights-file-input';
    weightsInput.multiple = true; // Use the boolean property, not setAttribute('multiple', 'true')

    // 4. Create the Load Button
    const loadButton = document.createElement('button');
    loadButton.id = 'load-button';
    loadButton.textContent = 'Load Model';

    // 5. Append the elements to the container (or directly to the body)
    loadModeldiv.appendChild(modelInput);
    loadModeldiv.appendChild(weightsInput);
    loadModeldiv.appendChild(loadButton);

    document.body.appendChild(loadModeldiv);



    
    trainBtn.addEventListener('click', async () => {
      await this.trainModel();
    });

    saveModelBtn.addEventListener('click', async () => {
      await this.model.save('downloads://my-model');
    });



  document.getElementById('load-button').addEventListener('click', async () => {
    const modelFileInput = document.getElementById('model-file-input');
    const weightsFileInput = document.getElementById('weights-file-input');

    // 1. Get the model.json File object
    const modelJsonFile = modelFileInput.files[0];

    // 2. Get the Weight File objects (an array)
    const weightFiles = Array.from(weightsFileInput.files);

    if (!modelJsonFile || weightFiles.length === 0) {
        console.error('Please select both the model.json and weight files.');
        return;
    }

    try {
        // 3. Create the input map required by tf.loadModel()
        // The structure needs to be: { [model.json file name]: model.json File object, ...weight files... }
        const filesMap = new Map();
        
        // Add the model.json file
        filesMap.set(modelJsonFile.name, modelJsonFile);
        
        // Add the weight files
        weightFiles.forEach(file => {
            filesMap.set(file.name, file);
        });

        // 4. Load the model using tf.loadModel()
        // Note: tf.loadModel() will accept the Map or an array of Files/Blobs.
        // Using the array of File objects is often simpler for local loading.
        const allFiles = [modelJsonFile, ...weightFiles];
        
        // This is the key part: pass the File objects directly to the function
        this.model = await tf.loadModel(tf.io.browserFiles(allFiles));

        console.log('✅ Model loaded successfully from local files!');
        console.log('Model Summary:', this.model.summary());
        this.modelTrained = true
        this.modelIsImported = true;
        trainBtn.innerText = 'Train new Model';

        
        // You can now use the 'model' object for inference (e.g., model.predict(...))
    } catch (error) {
        console.error('❌ Error loading model:', error);
    }
});

    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.mobilenet = await mobilenetModule.load();
    this.start();
  }


  ensureModel() {

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [this.embeddingSize],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(tf.layers.dense({
      units: NUM_CLASSES,
      activation: 'softmax',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
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


  addToDict(key, value) {
    if (!this.capturedDataset[key]) {
      this.capturedDataset[key] = [];   // create list if not exists
    }
    this.capturedDataset[key].push(value);
  }


  async test(image){

    let logits;

    // 'conv_preds' is the logits activation of MobileNet.
    const infer = () => this.mobilenet.infer(image, 'conv_preds');

    logits = infer();
    const emb = logits.as2D(1, -1);
    const preds = this.model.predict(emb);
    const probs = await preds.data();
    const classIndex = probs.indexOf(Math.max(...probs));

    return {probs, classIndex};

  }


  async animate() {

    if (this.videoPlaying) {


      const image = tf.fromPixels(this.video);

      // console.log("Video frame tensor shape:", image);
      let logits;

      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Capture examples if one of the buttons is held down
      if (this.training != -1) {

        // Draw the video frame to the canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Convert to image data or base64 image
        const dataURL = this.canvas.toDataURL("image/png");

        this.addToDict(this.training, dataURL);
        this.exampleCounts[this.training] += 1;

      }

      const totalExamples = this.exampleCounts.reduce((a, b) => a + b, 0);

      if (this.modelTrained) {

        // If the model is trained run test
        const {probs, classIndex} = await this.test(image);

        for (let i = 0; i < NUM_CLASSES; i++) {
          // Make the predicted class bold
          if (classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (this.exampleCounts[i] > 0) {
            const pct = (probs[i] * 100).toFixed(1);
            this.infoTexts[i].innerText = ` ${this.exampleCounts[i]} examples - ${pct}%`
          } else {
            this.infoTexts[i].innerText = ` 0 examples`;
          }
        }
      }
      else if (totalExamples > 0) {
        // Update example counts while collecting (no prediction yet)
        for (let i = 0; i < NUM_CLASSES; i++) {
          this.infoTexts[i].style.fontWeight = 'normal';
          this.infoTexts[i].innerText = ` ${this.exampleCounts[i]} examples`;
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }

    this.timer = requestAnimationFrame(this.animate.bind(this));
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
    
    // Optional: Preprocess the tensor (e.g., resize, normalize) for your model
    // const resizedTensor = tf.image.resizeBilinear(tensor, [targetHeight, targetWidth]);
    
    return tensor;
}


  async convertUrlToEmbedding(){
      // The outputed logits from mobilenet
      let logits;
      this.ensureModel();
      
      for (const key in this.capturedDataset) {

        for(const imageURL of this.capturedDataset[key]) {

          // Convert base64 image to tensor
          const image = await this.imageBase64ToTensor(imageURL);

          // 'conv_preds' is the logits activation of MobileNet.
          logits = this.mobilenet.infer(image, 'conv_preds');

          // Convert logits to 2D embedding
          const emb = logits.as2D(1, -1);

          // Store the embedding and the label
          this.trainXs.push(emb.clone());
          this.trainYs.push(tf.oneHot(tf.tensor1d([key]).toInt(), NUM_CLASSES));

          // Dispose tensors to free memory
          image.dispose();
          logits.dispose();
          emb.dispose();

        }
      }


  }

  async trainModel() {

    if (this.modelIsImported) {
      this.ensureModel();
    }

    this.trainStatus.innerText = ' Preparing data...';
    await this.convertUrlToEmbedding();

    if (this.trainXs.length === 0) {
      this.trainStatus.innerText = ' No examples to train on';
      return;
    }

    await tf.nextFrame();

    // Stack examples
    const xs = tf.concat(this.trainXs, 0);
    const ys = tf.concat(this.trainYs, 0);

    // Free per-example tensors
    this.trainXs.forEach(t => t.dispose());
    this.trainYs.forEach(t => t.dispose());
    this.trainXs = [];
    this.trainYs = [];

    this.trainStatus.innerText = ' Training...';

    const batchSize = Math.min(32, xs.shape[0]);
    const epochs = 20;

    await this.model.fit(xs, ys, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          this.trainStatus.innerText = ` Training epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(3)} acc: ${logs.acc !== undefined ? logs.acc.toFixed(3) : (logs.accuracy || 0).toFixed(3)}`;
          await tf.nextFrame();
        }
      }
    });

    xs.dispose();
    ys.dispose();

    this.modelTrained = true;
    this.trainStatus.innerText = ' Trained';
  }
}

window.addEventListener('load', () => new Main());