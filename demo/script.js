// Grab elements, create settings, etc.
let video = document.getElementById('video');
const images = [];


async function initialise() {
  var canvas = document.getElementById('canvas');
  var context = canvas.getContext('2d');
  var video = document.getElementById('video');


  //const model = await tf.loadLayersModel('file://model/model.json');
  const model = await tf.loadLayersModel('https://stivenmorvan.fr/projects/language_of_sign_recognition/demo/model/model.json');

  async function capture() {

      let input = tf.browser.fromPixels(video,1);
      // TODO: check Greyscale
      let originalShape = input.shape;
      input = tf.reshape(input,[1].concat(input.shape));

      let analysedImageLength = 300;
      centery = Math.floor(input.shape[1] / 2)
      centerx = Math.floor(input.shape[2] / 2)
      half = Math.floor(analysedImageLength / 2);
      x1 = centerx - half
      y1 = centery - half
      x2 = centerx + half
      y2 = centery + half

      input = tf.image.cropAndResize(input.asType('float32'), [[y1, x1, y2, x2]], [1], [64,64])

      const result = model.predict(input);
      result.print()
      console.log(result);

      input = tf.reshape(input,[64,64,1]);
      tf.browser.toPixels(input,canvas);


      // canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      // let image = new Image()
      // image.src = canvas.toDataURL();
      await tf.nextFrame();
  }
  // const result = await net.classify(imgEl);
  // console.log(result);

  // Trigger photo take
  document.getElementById("snap").addEventListener("click", function() {
    capture();
  });
}

// Get access to the camera!
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
