import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';

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
					window.X_ReportProgress(epoch, epochs, logs);
				},
				onTrainEnd: () => { 
					this.trainingStatus = 3;
					window.X_TrainingDone();
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
		if (!this.model) return null;

		let modelData = null;

		await this.model.save(tf.io.withSaveHandler(async (artifacts) => {
			modelData = artifacts;
			return { modelArtifactsInfo: artifacts.modelArtifactsInfo };
		}));

		// modelData now contains:
		// .modelTopology (the JSON structure)
		// .weightData (the binary weights)
		return modelData;
	}

	async importModel(modelTopology, weightSpecs, weightData) {
		const modelArtifacts = {
			modelTopology: modelTopology,
			weightSpecs: weightSpecs,
			weightData: weightData
		};

		// 3. Load the model
		this.model = await window.tf.loadLayersModel(window.tf.io.fromMemory(modelArtifacts));
		
		this.trainingStatus = 3;
		console.log("Model imported successfully from memory.");
		return true;
	}

	async test(source) {
		if (!this.model) return null;

		let imgElement = source;
		if (typeof source === 'string') {
			imgElement = await new Promise((resolve, reject) => {
					const img = new Image();
					img.onload = () => resolve(img);
					img.onerror = reject;
					img.src = source;
			});
		}

		return tf.tidy(() => {
			// Now imgElement is a valid HTMLImageElement
			const image = tf.fromPixels(imgElement);
			const logits = this.mobilenet.infer(image, 'conv_preds');
			const emb = logits.as2D(1, -1);
			const preds = this.model.predict(emb);
			
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
