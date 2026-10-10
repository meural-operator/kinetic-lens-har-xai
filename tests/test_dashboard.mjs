// Independent lifecycle checks: delayed permissions and old socket callbacks
// must never overwrite or stop a newer camera session. No physical camera used.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const elements = new Map();
function element() {
  return { value: '', checked:false, disabled: false, hidden: false, width:480, selectedOptions: [{disabled:false}],
    append() {}, replaceChildren() {}, dispatchEvent() {this.onchange?.();},
    getContext() {return {clearRect(){},fillRect(){},drawImage(){},fillText(){}};}, async play() {}, click() {} };
}
const document = {
  getElementById(id) {if (!elements.has(id)) elements.set(id,element());return elements.get(id);},
  createElement: element
};
let requests = [], sockets = [];
class Socket {
  static OPEN=1;
  readyState=0;
  constructor() {sockets.push(this);}
  close() {this.readyState=3;}
  send(message) {this.sent ??=[];this.sent.push(JSON.parse(message));}
}
const info={name:'Fixture',fps:5,window_seconds:4,update_seconds:1,max_frames:20,labels:['a','b'],methods:[{id:'temporal',available:true}],provenance:{runtime:'Fixture'}};
const context=vm.createContext({document, console, WebSocket:Socket, Image:class{},
  fetch:async()=>({ok:true,json:async()=>info}),
  navigator:{mediaDevices:{getUserMedia:()=>new Promise(resolve=>requests.push(resolve)),enumerateDevices:async()=>[]}},
  location:{protocol:'http:',host:'127.0.0.1:8765'},performance,
  window:{addEventListener(){}},Event:class{},
  setInterval:()=>1,clearInterval(){},
  setTimeout:(callback,ms)=>{const handle=setTimeout(callback,ms);handle.unref();return handle;}
});
vm.runInContext(fs.readFileSync('src/kinetic_lens/web/app.js','utf8'),context);
await new Promise(resolve=>setImmediate(resolve));
const media=()=>{const track={stops:0,stop(){this.stops++;},getSettings(){return {deviceId:'fixture'};}};return {track,getTracks:()=>[track],getVideoTracks:()=>[track]};};
const oldMedia=media(),newMedia=media();
const first=vm.runInContext('start()',context);
assert.equal(document.getElementById('stop').disabled,false);
vm.runInContext('stop()',context);
const second=vm.runInContext('start()',context);
requests[0](oldMedia);await first;
assert.ok(oldMedia.track.stops>0);
assert.equal(vm.runInContext('starting',context),true);
requests[1](newMedia);await second;
assert.equal(document.getElementById('video').srcObject,newMedia);
assert.equal(newMedia.track.stops,0);
const oldSocket=sockets[0];
vm.runInContext('stop()',context);
const thirdMedia=media(),third=vm.runInContext('start()',context);
requests[2](thirdMedia);await third;
oldSocket.onclose();
assert.equal(thirdMedia.track.stops,0);
assert.equal(document.getElementById('video').srcObject,thirdMedia);
// Optional scheduler never duplicates a window or queues work while explaining.
sockets.at(-1).readyState=1;
document.getElementById('method').value='temporal';
vm.runInContext('lastResult={window_end_ms:performance.now()};nextAutoAt=0',context);
assert.equal(vm.runInContext('requestExplanation(true)',context),true);
assert.equal(vm.runInContext('requestExplanation(true)',context),false);
assert.equal(sockets.at(-1).sent.length,1);
vm.runInContext('explaining=false;nextAutoAt=performance.now()+10000;lastResult={window_end_ms:performance.now()+1}',context);
assert.equal(vm.runInContext('requestExplanation(true)',context),false);
vm.runInContext('nextAutoAt=0',context);
assert.equal(vm.runInContext('requestExplanation(true)',context),true);
document.getElementById('auto-xai').checked=true;
vm.runInContext('receive({type:"error",message:"unsupported target"})',context);
assert.equal(document.getElementById('auto-xai').checked,false);
// Matching sample, rather than live video, is the overlay's image source.
const draws=[],fills=[];
document.getElementById('replay').getContext=()=>({clearRect(){},drawImage:im=>draws.push(im.id),fillRect:()=>fills.push(1)});
document.getElementById('heatmap').checked=true;
document.getElementById('opacity').value='45';
vm.runInContext('replayFrames=[{id:"frame-0",width:192,height:192},{id:"frame-1",width:192,height:192}];replayIndex=1;replayResult={target:0,data:{display_scale:1,maps:[[[0]],[[1]]]}};drawReplay()',context);
assert.deepEqual(draws,['frame-1']);
assert.equal(fills.length,2); // background + one attribution cell
document.getElementById('heatmap').checked=false;
vm.runInContext('drawReplay()',context);
assert.equal(fills.length,3); // only background when heatmap is off
vm.runInContext('stop()',context);
assert.ok(thirdMedia.track.stops>0);
assert.equal(document.getElementById('video').srcObject,null);
assert.equal(document.getElementById('replay-panel').hidden,true);
assert.equal(vm.runInContext('replayFrames',context),undefined);
console.log('Dashboard lifecycle, optional scheduler and matching-frame overlay checks passed.');
