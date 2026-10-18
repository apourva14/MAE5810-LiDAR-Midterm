const cloudSpec=__CLOUD_SPEC__;
const angularSpec=__ANGULAR_SPEC__;
const colourModes=__MODES__;
const returnTraces=__RETURN_META__;
const referenceTraces=__MESH_META__;
const rangeLabels=__RANGE_LABELS__;
const rangeOriginal=angularSpec.data[0].z;
const cloudDiv=document.getElementById('cloud-plot');
const angularDiv=document.getElementById('angular-plot');
const modeControl=document.getElementById('colour-mode');
const objectControl=document.getElementById('object-selection');
const groundControl=document.getElementById('show-ground');
const referenceControl=document.getElementById('show-reference');
const objectLegend=document.getElementById('object-legend');
const groundIDs=new Set(returnTraces.filter(t=>t.ground).map(t=>t.label));
const plotConfig={responsive:false,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',scale:2}};
function wanted(label,ground){
  if(ground)return groundControl.checked;
  return objectControl.value==='all'||String(label)===objectControl.value;
}
async function applyControls(){
  const key=modeControl.value, mode=colourModes[key];
  const indices=returnTraces.map(t=>t.index);
  if(indices.length){
    const colours=returnTraces.map(t=>cloudSpec.data[t.index].customdata.map(row=>
      key==='object'?t.label:(key==='snr'?Math.log10(Math.max(row[mode.field],1e-9)):row[mode.field])));
    await Plotly.restyle(cloudDiv,{visible:returnTraces.map(t=>wanted(t.label,t.ground)),'marker.color':colours},indices);
  }
  const layout={'coloraxis.colorscale':mode.scale,'coloraxis.cmin':mode.lo,'coloraxis.cmax':mode.hi,
    'coloraxis.showscale':key!=='object','coloraxis.colorbar.title.text':mode.title,
    'coloraxis.colorbar.tickmode':key==='object'?'array':'auto',
    'coloraxis.colorbar.tickvals':key==='object'?__TICKVALS__:null,
    'coloraxis.colorbar.ticktext':key==='object'?__TICKTEXT__:null};
  await Plotly.relayout(cloudDiv,layout);
  if(referenceTraces.length)await Plotly.restyle(cloudDiv,
    {visible:referenceTraces.map(t=>referenceControl.checked&&wanted(t.label,t.ground))},referenceTraces.map(t=>t.index));
  const z=rangeOriginal.map((row,r)=>row.map((value,c)=>{
    const label=rangeLabels[r][c];
    return label>=0&&wanted(label,groundIDs.has(label))?value:null;
  }));
  await Plotly.restyle(angularDiv,{z:[z]},[0]);
  objectLegend.hidden=key!=='object';
  objectLegend.querySelectorAll('.legend-item').forEach(item=>
    item.classList.toggle('is-muted',!wanted(Number(item.dataset.label),item.dataset.ground==='1')));
  document.getElementById('colour-explanation').textContent=mode.explanation;
  await sizePlots();
}
let cameraIsNarrow=null;
async function sizePlots(){
  const mobile=window.innerWidth<600, categorical=modeControl.value==='object';
  const layout={
    'margin.l':mobile?54:65,'margin.r':categorical?(mobile?54:24):(mobile?100:120),
    'margin.t':18,'margin.b':24,
    'scene.xaxis.title.text':mobile?'x (m)':'World x (m)',
    'scene.yaxis.title.text':mobile?'y (m)':'World y (m)',
    'scene.zaxis.title.text':mobile?'z (m)':'World z (m)'};
  if(cameraIsNarrow!==mobile){
    layout['scene.camera.eye']=mobile?{x:-2.8,y:-2.5,z:1.6}:{x:-1.7,y:-1.5,z:1.0};
    cameraIsNarrow=mobile;
  }
  await Plotly.relayout(cloudDiv,layout);
  await Promise.all([Plotly.Plots.resize(cloudDiv),Plotly.Plots.resize(angularDiv)]);
}
function reportError(error){
  document.getElementById('colour-explanation').textContent='The plot could not be displayed: '+error.message;
}
let viewUpdates=Promise.resolve();
function queueUpdate(resizeOnly=false){
  document.documentElement.dataset.viewerReady='false';
  viewUpdates=viewUpdates.catch(reportError).then(()=>resizeOnly?sizePlots():applyControls());
  const current=viewUpdates;
  current.then(()=>{
    if(current===viewUpdates)document.documentElement.dataset.viewerReady='true';
  },reportError);
  return current;
}
cloudDiv.innerHTML='';
Promise.all([Plotly.newPlot(cloudDiv,cloudSpec.data,cloudSpec.layout,plotConfig),
             Plotly.newPlot(angularDiv,angularSpec.data,angularSpec.layout,plotConfig)])
.then(async()=>{
  await queueUpdate();
  document.getElementById('display-controls').disabled=false;
  [modeControl,objectControl,groundControl,referenceControl].forEach(control=>
    control.addEventListener('change',()=>queueUpdate()));
  let resizeTimer;
  window.addEventListener('resize',()=>{
    clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>queueUpdate(true),100);
  });
}).catch(reportError);
