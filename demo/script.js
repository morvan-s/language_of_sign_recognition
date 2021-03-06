let canvas = document.getElementById('canvas');
let video = document.getElementById('video');
let result = document.getElementById('result');
let cropLength = 200;
let model_url = 'https://raw.githubusercontent.com/morvan-s/language_of_sign_recognition/master/demo/models/model/model.json';

const sleep = m => new Promise(r => setTimeout(r, m));

async function initialise() {
  const model = await tf.loadLayersModel(model_url);

  let input = tf.browser.fromPixels(video,1);
  let centery = Math.floor(input.shape[0] / 2);
  let centerx = Math.floor(input.shape[1] / 2);
  let half = Math.floor(cropLength / 2);
  let x1 = centerx - half;
  let y1 = centery - half;
  let x2 = centerx + half;
  let y2 = centery + half;

  let boxes = [[y1/input.shape[0], x1/input.shape[1], y2/input.shape[0], x2/input.shape[1]]]

  while(true){
    input = input.div(tf.scalar(255));
    input = tf.reshape(input,[1].concat(input.shape));

    input = tf.image.cropAndResize(input.asType('float32'), boxes, [0], [64,64]);

    const res = (model.predict(input)).arraySync()[0];
    let i = res.indexOf(Math.max(...res));
    console.log(res);
    (res[i] >= 0.4) ? result.textContent=i : result.textContent="NaN";
    input = tf.reshape(input,[64,64,1]);
    tf.browser.toPixels(input,canvas);

    await sleep(700);
    await tf.nextFrame();
    input = tf.browser.fromPixels(video,1);
  }
}

// Initialise the camera and the detection
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        video.srcObject = stream;
        video.play();
        initialise();
    });
}
