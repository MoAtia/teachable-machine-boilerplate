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
const NUM_CLASSES = 4;
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
    this.embeddingSize = null;

    // Initiate the page (load mobilenet, etc.)
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

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

    trainBtn.addEventListener('click', async () => {
      await this.trainModel();
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

    // Build the classifier model (two dense layers)
    this.buildModel();

    this.start();
  }

  buildModel() {
    // We will determine input size after first embedding; for now create a placeholder model
    // and rebuild when we know embedding size.
    this.model = null;
  }

  ensureModel(embeddingSize) {
    if (this.model && this.embeddingSize === embeddingSize) {
      return;
    }
    this.embeddingSize = embeddingSize;
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

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Capture examples if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();
        const emb = logits.as2D(1, -1);
        const size = emb.shape[1];
        this.ensureModel(size);
        // Store example and label
        this.trainXs.push(emb.clone());
        this.trainYs.push(tf.oneHot(tf.tensor1d([this.training]).toInt(), NUM_CLASSES));
        this.exampleCounts[this.training] += 1;
      }

      const totalExamples = this.exampleCounts.reduce((a, b) => a + b, 0);
      if (this.modelTrained) {

        // If the model is trained run predict
        logits = infer();
        const emb = logits.as2D(1, -1);
        const preds = this.model.predict(emb);
        const probs = await preds.data();
        const classIndex = probs.indexOf(Math.max(...probs));
        preds.dispose();
        emb.dispose();

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
      } else if (totalExamples > 0) {
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

  async trainModel() {
    if (this.trainXs.length === 0) {
      this.trainStatus.innerText = ' No examples to train on';
      return;
    }
    this.trainStatus.innerText = ' Preparing data...';
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