$(document).ready(function () {

    var carousels = bulmaCarousel.attach('#results-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    });

    // Start playing next video in carousel and pause previous video to limit load on browser
    for (var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            var nextId = (state.next + state.length) % state.length;  // state.next can be -1 or larger than the number of videos
            var nextVideoElement = $("#results-carousel .slider-item[data-slider-index='" + nextId + "'] video")[0];
            var previousVideoElement = $("#results-carousel .slider-item[data-slider-index='" + state.index + "'] video")[0];

            previousVideoElement.pause();
            previousVideoElement.currentTime = 0;
            nextVideoElement.currentTime = 0;
            nextVideoElement.play();
        });
    }

    var reenactmentsCarousels = bulmaCarousel.attach('#reenactments-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    });

    // Start playing next video in carousel and pause previous video to limit load on browser
    for (var i = 0; i < reenactmentsCarousels.length; i++) {
        // Add listener to  event
        reenactmentsCarousels[i].on('before:show', state => {
            var nextId = (state.next + state.length) % state.length;  // state.next can be -1 or larger than the number of videos
            var nextVideoElement = $("#reenactments-carousel .slider-item[data-slider-index='" + nextId + "'] video")[0];
            var previousVideoElement = $("#reenactments-carousel .slider-item[data-slider-index='" + state.index + "'] video")[0];

            previousVideoElement.pause();
            previousVideoElement.currentTime = 0;
            nextVideoElement.currentTime = 0;
            nextVideoElement.play();
        });
    }


})

// From https://dorverbin.github.io/refnerf/.
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth / 2;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");


    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
            // videoMerge.setAttribute('data-position', position);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);

        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            requestAnimationFrame(drawLoop);

            // // Draw border
            // mergeContext.beginPath();
            // mergeContext.moveTo(vidWidth * position, 0);
            // mergeContext.lineTo(vidWidth * position, vidHeight);
            // mergeContext.closePath()
            // mergeContext.strokeStyle = "#444444";
            // mergeContext.lineWidth = 5;
            // mergeContext.stroke();
        }
        requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};


function resizeAndPlay(element) {
    var cv = document.getElementById(element.id + "Merge");
    cv.width = element.videoWidth / 2;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it

    playVids(element.id);
}

document.addEventListener('DOMContentLoaded', (event) => {
    var videoMerge = document.getElementById('geo1Merge');
    var arrowIcon = document.getElementById('dragArrow');

    if (videoMerge && arrowIcon) {
        var hasInteracted = false; // Flag to check if the user has interacted

        function hideArrowOnInteraction() {
            if (!hasInteracted) {
                arrowIcon.style.opacity = 0;
                hasInteracted = true; // Set the flag to true after first interaction
            }
        }

        videoMerge.addEventListener('mousemove', hideArrowOnInteraction);
        videoMerge.addEventListener('touchmove', hideArrowOnInteraction);
    } else {
        console.error('Required elements not found');
    }
});
