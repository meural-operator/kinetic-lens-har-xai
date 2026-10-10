// Independent lifecycle checks: delayed permissions and old socket callbacks
// must never overwrite or stop a newer camera session. No physical camera used.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const elements = new Map();
function element() {
  return { value: '', disabled: false, hidden: false, selectedOptions: [{disabled:false}],
    append() {}, replaceChildren() {}, dispatchEvent() {this.onchange?.();},
    getContext() {return {};}, async play() {}, click() {} };
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
}
const info={name:'Fixture',fps:5,window_seconds:4,max_frames:20,labels:['a','b'],methods:[{id:'temporal',available:true}],provenance:{runtime:'Fixture'}};
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
vm.runInContext('stop()',context);
assert.ok(thirdMedia.track.stops>0);
assert.equal(document.getElementById('video').srcObject,null);
console.log('Dashboard delayed-permission and reconnect lifecycle checks passed.');
