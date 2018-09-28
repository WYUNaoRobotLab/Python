function sendMessage(content,avatar){
    var tpl=" <div class=\"right-message-wrapper\"> <div class=\"right-message-bubble\"> <div class=\"right-message-container-wrapper\"> <div class=\"message-container\"> <p> "+content+"</p> </div> </div> </div> <div class=\"avatar-wrapper\"> <img class=\"avatar\" src=\"http://www.jmtung.cn/images/icon.jpg\" alt=\"avatar\"> </div> </div>"
    document.getElementById("dialog").innerHTML+=tpl;
    window.scrollTo(0,document.body.scrollHeight);
}

function getMessage(content,avatar) {
    var tpl="<div class=\"left-message-wrapper\"> <div class=\"avatar-wrapper\"> <img class=\"avatar\" src=\"http://www.jmtung.cn/images/icon.jpg\" alt=\"avatar\"> </div> <div class=\"left-message-bubble\"> <div class=\"left-message-container-wrapper\"> <div class=\"message-container\"> <p>  "+content+" </p> </div> </div> </div> </div>"
    document.getElementById("dialog").innerHTML+=tpl;
    window.scrollTo(0,document.body.scrollHeight);
}

function sendImg(url,avatar) {
    var tpl=" <div class=\"right-message-wrapper\"> <div class=\"right-img-wrapper\"> <img class=\"message-img\" src=\""+url+"\" onclick=\"show('"+url+"')\"> </div> <div class=\"avatar-wrapper\"> <img class=\"avatar\" src=\"http://www.jmtung.cn/images/icon.jpg\" alt=\"avatar\"> </div> </div>";
    document.getElementById("dialog").innerHTML+=tpl;
    window.scrollTo(0,document.body.scrollHeight);
}

function getImg(url,avatar) {
    var tpl=" <div class=\"left-message-wrapper\"> <div class=\"avatar-wrapper\"> <img class=\"avatar\" src=\"http://www.jmtung.cn/images/icon.jpg\" alt=\"avatar\"> </div> <div class=\"left-img-wrapper\"> <img class=\"message-img\" src=\""+url+"\" onclick=\"show('"+url+"')\"> </div> </div>"
    document.getElementById("dialog").innerHTML+=tpl;
    window.scrollTo(0,document.body.scrollHeight);
}

function show(url) {
    document.getElementById("large-img-mask").getElementsByTagName('img')[0].src=url;
    document.getElementById("large-img-mask").style.display='flex';
}

function mClose() {
    document.getElementById("large-img-mask").style.display="none";
}