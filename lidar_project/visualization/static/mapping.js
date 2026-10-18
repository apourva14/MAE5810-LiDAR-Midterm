const $=id=>document.getElementById(id), C=MAPDATA.colors;
let timeline=[],index=0,playing=false,timer=null,queue=Promise.resolve(),revision=0;
const defaultCamera={eye:{x:-1.25,y:-1.55,z:1.25},up:{x:0,y:0,z:1}};
const statusName=s=>({goal:'Goal reached',collision:'Collision',timeout:'Time limit'}[s]||s);
const alpha=(color,a)=>/^#[0-9a-f]{6}$/i.test(color)?`rgba(${parseInt(color.slice(1,3),16)},${parseInt(color.slice(3,5),16)},${parseInt(color.slice(5,7),16)},${a})`:color;
MAPDATA.cases.forEach(c=>{for(const id of ['environment-a','environment-b']){const option=document.createElement('option');option.value=c.id;option.textContent=c.label;$(id).appendChild(option);}});
$('environment-a').value=MAPDATA.cases[0].id;$('environment-b').value=MAPDATA.cases[Math.min(1,MAPDATA.cases.length-1)].id;
for(const name of ['unknown','low','surface'])document.documentElement.style.setProperty('--'+name,C[name]);
const active=()=>[$('environment-a').value,...($('view-mode').value==='compare'?[$('environment-b').value]:[])].map(id=>MAPDATA.cases.find(c=>c.id===id));
function pause(){playing=false;clearTimeout(timer);$('play').textContent='Play';}
function resetSelection(){const t=timeline[index]||0;pause();const cs=active();timeline=[...new Set(cs.flatMap(c=>c.states.t.map(t=>+t.toFixed(8))))].sort((a,b)=>a-b);const next=timeline.findIndex(x=>x>=t);index=next<0?timeline.length-1:next;$('timeline').max=timeline.length-1;$('case-b').hidden=cs.length===1;$('environment-b-field').hidden=cs.length===1;$('environment-panels').classList.toggle('single',cs.length===1);revision++;enqueue();}
function schedule(){if(!playing)return;if(index>=timeline.length-1){pause();return;}const dt=timeline[index+1]-timeline[index];timer=setTimeout(()=>{index++;enqueue().then(schedule);},Math.max(25,1000*dt/(+$('speed').value)));}
$('play').addEventListener('click',()=>{if(playing)return pause();if(index>=timeline.length-1)index=0;playing=true;$('play').textContent='Pause';enqueue().then(schedule);});
$('previous').addEventListener('click',()=>{pause();index=Math.max(0,index-1);enqueue();});
$('next').addEventListener('click',()=>{pause();index=Math.min(timeline.length-1,index+1);enqueue();});
$('timeline').addEventListener('input',()=>{pause();index=+$('timeline').value;enqueue();});
$('speed').addEventListener('change',()=>{if(playing){clearTimeout(timer);schedule();}});
['environment-a','environment-b','view-mode'].forEach(id=>$(id).addEventListener('change',resetSelection));
['show-map','show-scan','show-low','show-truth','show-route','follow'].forEach(id=>$(id).addEventListener('change',()=>enqueue()));
$('reset-view').addEventListener('click',()=>{revision++;enqueue();});
function fail(e){pause();$('error').hidden=false;$('error').textContent='The view could not be displayed: '+e.message;}
function enqueue(){const t=timeline[index]||0;document.documentElement.dataset.viewerReady='false';queue=queue.catch(fail).then(()=>draw(t));return queue;}
function stateIndex(times,t){let lo=0,hi=times.length;while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<=t+1e-8)lo=mid+1;else hi=mid;}return Math.max(0,lo-1);}
function cylinder(x,y,r,h){const X=[],Y=[],Z=[],I=[],J=[],K=[],n=24;for(const z of [0,h])for(let i=0;i<n;i++){const a=2*Math.PI*i/n;X.push(x+r*Math.cos(a));Y.push(y+r*Math.sin(a));Z.push(z);}X.push(x,x);Y.push(y,y);Z.push(0,h);for(let i=0;i<n;i++){const j=(i+1)%n;for(const f of [[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]]){I.push(f[0]);J.push(f[1]);K.push(f[2]);}}return{x:X,y:Y,z:Z,i:I,j:J,k:K,type:'mesh3d',color:C.robot,opacity:.9,flatshading:true,lighting:{ambient:.9,diffuse:.3},name:'Robot body',showlegend:false,hoverinfo:'name'};}
function circle(x,y,r,z,n=40){const X=[],Y=[],Z=[];for(let i=0;i<=n;i++){const a=2*Math.PI*i/n;X.push(x+r*Math.cos(a));Y.push(y+r*Math.sin(a));Z.push(z);}return{x:X,y:Y,z:Z};}
function gridAt(c,step){let observed=0;const z=c.ground_first.map((row,r)=>row.map((first,col)=>{const surface=c.surface_first[r][col],v=surface>=0&&surface<=step?2:first>=0&&first<=step?1:0;if(v)observed++;return v;}));return{z,observed};}
function lineTrace(x,y,z,color,width,name){return{x,y,z,type:'scatter3d',mode:'lines',line:{color,width},name,showlegend:false,hoverinfo:'skip'};}
async function drawCase(c,slot,t){
 const k=stateIndex(c.states.t,t),scanStep=Math.min(k,c.scans.length-1),step=scanStep,ended=k===c.states.t.length-1;
 const xy=c.states.xy[k],theta=c.states.theta[k],scan=scanStep>=0?c.scans[scanStep]:{x:[],y:[],z:[],intensity:[]},scanTime=scanStep>=0?c.states.t[scanStep]:null;
 $('label-'+slot).textContent=c.label;$('description-'+slot).textContent=c.description;
 $('state-'+slot).textContent=`${ended?statusName(c.status)+'; holding final pose':'Moving'} | robot t = ${c.states.t[k].toFixed(1)} s | ${ended?'Last scan':'Scan'} t = ${scanTime==null?'none':scanTime.toFixed(1)+' s'}`;
 const count=scanStep>=0?c.point_counts[scanStep]:0;
 $('counts-'+slot).textContent=`${count.toLocaleString()} map voxels | ${scanStep>=0?c.scan_counts[scanStep].toLocaleString():0} total scan returns; ${$('show-scan').checked?scan.x.length:0} shown in mapped area | seed ${c.seed}`;
 const traces=[];
 if($('show-map').checked){const cloud=c.cloud,ix=[];for(let i=0;i<cloud.step.length;i++)if(cloud.step[i]<=step&&($('show-low').checked||cloud.z[i]>c.ground_threshold))ix.push(i);traces.push({type:'scatter3d',mode:'markers',x:ix.map(i=>cloud.x[i]),y:ix.map(i=>cloud.y[i]),z:ix.map(i=>cloud.z[i]),customdata:ix.map(i=>[cloud.intensity[i],cloud.step[i]]),marker:{size:.65,opacity:.65,color:ix.map(i=>cloud.z[i]),colorscale:MAPDATA.height_scale,cmin:MAPDATA.z_limits[0],cmax:MAPDATA.z_limits[1],colorbar:{title:{text:'Height (m)',font:{size:10}},thickness:9,len:.65,tickfont:{size:9},outlinewidth:0}},name:'Measured map',showlegend:true,hovertemplate:'x=%{x:.2f} m · y=%{y:.2f} m<br>z=%{z:.2f} m · intensity=%{customdata[0]:.3f}<br>First observed at step %{customdata[1]}<extra>measured map voxel</extra>'});}
 if($('show-scan').checked){traces.push({type:'scatter3d',mode:'markers',x:scan.x,y:scan.y,z:scan.z,customdata:scan.intensity,marker:{size:1.0,color:C.scan,opacity:.95},name:ended?'Last recorded scan':'Current scan',hovertemplate:'x=%{x:.2f} m · y=%{y:.2f} m<br>z=%{z:.2f} m · intensity=%{customdata:.3f}<extra>recorded 3D return</extra>'});}
 if($('show-truth').checked)for(const mesh of c.meshes)traces.push({type:'mesh3d',x:mesh.vertices.map(v=>v[0]),y:mesh.vertices.map(v=>v[1]),z:mesh.vertices.map(v=>v[2]),i:mesh.faces.map(f=>f[0]),j:mesh.faces.map(f=>f[1]),k:mesh.faces.map(f=>f[2]),color:mesh.unknown?C.unexpected:C.truth,opacity:.15,flatshading:true,name:mesh.name,showlegend:false,hovertemplate:mesh.name+'<extra>ground-truth reference scene</extra>'});
 if($('show-route').checked)traces.push({...lineTrace(c.reference.map(p=>p[0]),c.reference.map(p=>p[1]),c.reference.map(()=>.025),C.truth,2,'Planned route'),line:{color:C.truth,width:2,dash:'dash'}});
 traces.push(lineTrace(c.states.xy.slice(0,k+1).map(p=>p[0]),c.states.xy.slice(0,k+1).map(p=>p[1]),c.states.xy.slice(0,k+1).map(()=>.035),C.path,3,'Executed path'));
 traces.push(cylinder(xy[0],xy[1],c.radius,c.body_height));
 traces.push(lineTrace([xy[0],xy[0]],[xy[1],xy[1]],[c.body_height,c.sensor_height],C.robot,4,'Sensor mast'));
 traces.push({type:'scatter3d',mode:'markers',x:[xy[0]],y:[xy[1]],z:[c.sensor_height],marker:{size:5,color:C.sensor,symbol:'diamond'},name:'LiDAR sensor',showlegend:true,hovertemplate:'Robot sensor position<br>x=%{x:.2f} m · y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>'});
 traces.push(lineTrace([xy[0],xy[0]+.9*Math.cos(theta)],[xy[1],xy[1]+.9*Math.sin(theta)],[c.body_height+.04,c.body_height+.04],C.sensor,5,'Robot heading'));
 const upper=circle(xy[0],xy[1],c.radius,c.envelope_height),lower=circle(xy[0],xy[1],c.radius,0);
 const envelope={x:[...lower.x,null,...upper.x],y:[...lower.y,null,...upper.y],z:[...lower.z,null,...upper.z]};
 for(const angle of [0,Math.PI/2,Math.PI,3*Math.PI/2]){const x=xy[0]+c.radius*Math.cos(angle),y=xy[1]+c.radius*Math.sin(angle);envelope.x.push(null,x,x);envelope.y.push(null,y,y);envelope.z.push(null,0,c.envelope_height);}
 traces.push({...lineTrace(envelope.x,envelope.y,envelope.z,C.robot,1,'Collision envelope'),opacity:.2});
 if(ended&&scanStep>=0&&$('show-scan').checked){const origin=c.states.xy[scanStep];traces.push({type:'scatter3d',mode:'markers',x:[origin[0]],y:[origin[1]],z:[c.sensor_height],marker:{size:5,color:C.scan,symbol:'diamond-open'},name:'Last scan origin',showlegend:false,hovertemplate:`Original scan origin at t=${scanTime.toFixed(1)} s<extra></extra>`});}
 let xr=c.extent.slice(0,2),yr=c.extent.slice(2,4);if($('follow').checked){xr=[xy[0]-5,xy[0]+5];yr=[xy[1]-5,xy[1]+5];}
 const axis={gridcolor:C.grid,zerolinecolor:C.grid,showbackground:true,backgroundcolor:C.background,tickfont:{size:9},title:{font:{size:10}}};
 const scene={aspectmode:'data',dragmode:'orbit',camera:defaultCamera,uirevision:c.id+'-'+revision,xaxis:{...axis,title:{text:'World x (m)'},range:xr},yaxis:{...axis,title:{text:'World y (m)'},range:yr},zaxis:{...axis,title:{text:'World z (m)'},range:MAPDATA.z_limits}};
 const layout={width:$('cloud-'+slot).clientWidth,height:$('cloud-'+slot).clientHeight,paper_bgcolor:C.background,font:{family:MAPDATA.font,color:C.text,size:11},scene,margin:{l:25,r:45,t:0,b:35},legend:{orientation:'h',x:0,y:-.02,font:{size:10}},uirevision:c.id+'-'+revision};
 const observed=gridAt(c,step),fraction=observed.observed/c.map_cells;
 $('area-'+slot).textContent=`${(100*fraction).toFixed(1)}% of map cells observed`;
 const scale=[[0,C.unknown],[1/3,C.unknown],[1/3,C.low],[2/3,C.low],[2/3,C.surface],[1,C.surface]];
 const grid=[{type:'heatmap',z:observed.z,x0:c.extent[0]+c.resolution/2,dx:c.resolution,y0:c.extent[2]+c.resolution/2,dy:c.resolution,zmin:-.5,zmax:2.5,colorscale:scale,showscale:false,hovertemplate:'x=%{x:.1f} m · y=%{y:.1f} m<br>Endpoint class %{z} (0 unknown, 1 low, 2 elevated)<extra></extra>'}];
 const trail=c.states.xy.slice(0,k+1);
 grid.push({type:'scatter',mode:'lines',x:trail.map(p=>p[0]),y:trail.map(p=>p[1]),line:{color:C.path,width:2},showlegend:false,hoverinfo:'skip'});
 if($('show-route').checked)grid.push({type:'scatter',mode:'lines',x:c.reference.map(p=>p[0]),y:c.reference.map(p=>p[1]),line:{color:C.background,width:1.2,dash:'dash'},showlegend:false,hoverinfo:'skip'});
 const ring=circle(xy[0],xy[1],c.radius,0);grid.push({type:'scatter',mode:'lines',x:ring.x,y:ring.y,line:{color:C.robot,width:1.6},showlegend:false,hoverinfo:'skip'});
 const gridLayout={width:$('grid-'+slot).clientWidth,height:$('grid-'+slot).clientHeight,paper_bgcolor:C.background,plot_bgcolor:C.background,font:{family:MAPDATA.font,color:C.text,size:10},margin:{l:46,r:16,t:10,b:40},xaxis:{title:{text:'World x (m)'},range:c.extent.slice(0,2),constrain:'domain',zeroline:false},yaxis:{title:{text:'World y (m)'},range:c.extent.slice(2,4),constrain:'domain',scaleanchor:'x',scaleratio:1,zeroline:false},uirevision:'grid-'+c.id+'-'+revision};
 await Promise.all([Plotly.react('cloud-'+slot,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true}),Plotly.react('grid-'+slot,grid,gridLayout,{responsive:true,displaylogo:false,scrollZoom:true})]);
}
async function draw(t){$('timeline').value=index;$('time-label').textContent=t.toFixed(1)+' s';await Promise.all(active().map((c,i)=>drawCase(c,i?'b':'a',t)));document.documentElement.dataset.viewerReady='true';}
const runCounts=[...new Set(MAPDATA.cases.map(c=>c.aggregate.runs??1))],seeds=[...new Set(MAPDATA.cases.map(c=>c.seed))];
$('comparison-description').textContent=`Motion statistics include all saved seeds${runCounts.length===1?' ('+runCounts[0]+' per environment)':''}; all map statistics use ${seeds.length===1?'the representative seed-'+seeds[0]+' maps':'the representative runs shown above'}. End times include failures.`;
for(const c of MAPDATA.cases){const a=c.aggregate,m=c.metrics,row=document.createElement('tr');const values=[c.label,m.n_obstacles,`${a.goals??(c.status==='goal'?1:0)}/${a.runs??1}`,(a.mean_elapsed_s??m.elapsed_s).toFixed(1),(a.mean_path_length_m??m.path_length_m).toFixed(1),(a.worst_surface_gap_m??m.swept_min_surface_gap_m).toFixed(2),(100*m.observed_fraction).toFixed(1)+'%',m.map_voxels.toLocaleString()];for(const value of values){const td=document.createElement('td');td.textContent=value;row.appendChild(td);}$('results-body').appendChild(row);}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>enqueue(),100);});
resetSelection();
