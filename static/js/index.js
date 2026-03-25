document.addEventListener("DOMContentLoaded", (event) => {
    var b = document.querySelectorAll('.b-dics');
    b.forEach(element =>
        new Dics({
            container: element,
            textPosition: 'bottom',
            arrayBackgroundColorText: ['#000000', '#000000', '#000000'],
            arrayColorText: ['#FFFFFF', '#FFFFFF', '#FFFFFF'],
            linesColor: '#ffffff'
        })
    );

});

function CopyToClipboard(id) {
    var r = document.createRange();
    r.selectNode(document.getElementById(id));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(r);
    document.execCommand('copy');
    window.getSelection().removeAllRanges();
}

function copytoclip() {
    // retire clipboard
    document.getElementById('checkbox1').classList.remove('is-hidden');
    document.getElementById('clipboard1').classList.add('is-hidden');
    document.getElementById('clipboard2').classList.add('is-hidden');

    document.getElementById('clip-copy').blur();

    setTimeout(function () {
        document.getElementById('checkbox1').classList.add('is-hidden');
        document.getElementById('clipboard1').classList.remove('is-hidden');
        document.getElementById('clipboard2').classList.remove('is-hidden');
    }, 2000);
    CopyToClipboard('citation_text');
}