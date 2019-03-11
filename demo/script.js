var canvas = document.getElementById('canvas');
var video = document.getElementById('video');
var result = document.getElementById('result');

const sleep = m => new Promise(r => setTimeout(r, m));

async function initialise() {

  const model = await tf.loadLayersModel('https://stivenmorvan.fr/projects/language_of_sign_recognition/demo/model/model.json');

  async function capture() {
      while(true){
        let input = tf.browser.fromPixels(video,1);
        let originalShape = input.shape;
        input = input.div(tf.scalar(255))
        input = tf.reshape(input,[1].concat(input.shape));

        let analysedImageLength = 200;
        centery = Math.floor(input.shape[1] / 2)
        centerx = Math.floor(input.shape[2] / 2)
        half = Math.floor(analysedImageLength / 2);
        x1 = centerx - half
        y1 = centery - half
        x2 = centerx + half
        y2 = centery + half

        input = tf.image.cropAndResize(input.asType('float32'), [[y1/input.shape[1], x1/input.shape[2], y2/input.shape[1], x2/input.shape[2]]], [0], [64,64])

        const res = (model.predict(input)).arraySync()[0];
        console.log(res);
        let i = res.indexOf(Math.max(...res));
        console.log(i);
        if(res[i] >= 0.4){
          result.textContent=i;
        } else {
          result.textContent="NaN";
        }
        input = tf.reshape(input,[64,64,1]);
        tf.browser.toPixels(input,canvas);

        await sleep(700);
        await tf.nextFrame();
      }
  }

  capture();
}

// Initialise the camera and the detection
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
        initialise();
    });
} else if(navigator.getUserMedia) { // Standard
    navigator.getUserMedia({ video: true }, function(stream) {
        video.src = stream;
        video.play();
        initialise();
    }, errBack);
} else if(navigator.webkitGetUserMedia) { // WebKit-prefixed
    navigator.webkitGetUserMedia({ video: true }, function(stream){
        video.src = window.webkitURL.createObjectURL(stream);
        video.play();
        initialise();
    }, errBack);
} else if(navigator.mozGetUserMedia) { // Mozilla-prefixed
    navigator.mozGetUserMedia({ video: true }, function(stream){
        video.srcObject = stream;
        video.play();
        initialise();
    }, errBack);
}
