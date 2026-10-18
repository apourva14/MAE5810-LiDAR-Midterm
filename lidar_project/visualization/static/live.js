let meta=null, run=-1, shown=-1, busy=false;
const $=id=>document.getElementById(id);
const j=async u=>(await fetch(u)).json();
const circ=(cx,cy,r,n=40)=>{const X=[],Y=[];for(let k=0;k<=n;k++){const a=2*Math.PI*k/n;X.push(cx+r*Math.cos(a));Y.push(cy+r*Math.sin(a));}return [X,Y];};
const live=()=>$('live').checked;

async function tick(){
  if(busy) return; busy=true;
  try{
    const m=await j('/meta');
    if(m.run!==run && m.n>0){run=m.run; shown=-1; $('live').checked=true;}
    if(m.objects) meta=m;
    if(meta && m.run===run){
      meta.n=m.n; const sl=$('sl'); sl.max=Math.max(m.n-1,0);
      const i=live()?m.n-1:+sl.value;
      if(live()) sl.value=i;
      if(m.n>0 && i!==shown){const f=await j('/frame?i='+i); draw(f); shown=f.i; $('fi').textContent=f.i+' / '+(m.n-1);}
    }
  }catch(e){}
  busy=false;
}
setInterval(tick,150);
$('sl').addEventListener('input',()=>{$('live').checked=false;tick();});
['live','lidar','mapon','mode'].forEach(id=>$(id).addEventListener('change',()=>{shown=-1;tick();}));

function draw(f){
  const R=meta.radius, W=meta.cam_w, H=meta.cam_h, lid=$('lidar').checked, mode=$('mode').value;
  $('title').textContent=`scenario '${meta.name}'   alpha = ${meta.alpha.toPrecision(3)} 1/m   R_max(rho=0.1) = ${meta.rmax.toFixed(1)} m`;
  $('hud').textContent=`t = ${f.t.toFixed(1)} s   v = ${f.v.toFixed(2)} m/s   omega = ${(f.w*180/Math.PI).toFixed(1)} deg/s   `+
    `true gap = ${f.dtrue==null?'none':(f.dtrue-R).toFixed(2)} m   perceived gap = ${f.dperc==null?'none':(f.dperc-R).toFixed(2)} m   `+
    `returns = ${f.n}   obstacle pts = ${f.n_obs}   ${f.status}`;

  // ---------- first-person view ----------
  const camTr=lid?[{x:f.cam.u,y:f.cam.v,type:'scatter',mode:'markers',
    marker:{size:f.cam.r.map(r=>Math.min(10,Math.max(3,40/Math.max(r,1)))),color:f.cam.r,colorscale:THEME.colormaps.camera_returns,cmin:0,
      cmax:Math.min(meta.rmax,30),colorbar:{title:'LiDAR range [m]',thickness:10,len:0.6}},
    hovertemplate:'range %{marker.color:.1f} m<extra></extra>'}]:[];
  Plotly.react('cam',camTr,{font:{family:THEME.font_family},
    title:{text:'first-person view (render + LiDAR returns)',font:{size:13}},
    images:[{source:f.img,xref:'x',yref:'y',x:0,y:0,sizex:W,sizey:H,xanchor:'left',yanchor:'top',sizing:'stretch',layer:'below'}],
    xaxis:{range:[0,W],visible:false,constrain:'domain'},
    yaxis:{range:[H,0],visible:false,scaleanchor:'x',constrain:'domain'},
    margin:{l:10,r:10,t:40,b:10},showlegend:false,uirevision:'cam'+run,
    annotations:[{xref:'paper',yref:'paper',x:0.01,y:0.01,xanchor:'left',yanchor:'bottom',showarrow:false,
      font:{color:THEME.colors.white,size:11},bgcolor:THEME.colors.overlay,
      text:`nearest perceived obstacle: ${f.dperc==null?'none':f.dperc.toFixed(2)} m   visibility ~ ${Math.min(3.9/meta.alpha,9999).toFixed(0)} m`}]
  },{responsive:true,displaylogo:false});

  // ---------- whole map ----------
  const tr=[];
  if($('mapon').checked) tr.push({type:'heatmap',z:f.grid,x0:meta.extent[0]+meta.res/2,dx:meta.res,y0:meta.extent[2]+meta.res/2,dy:meta.res,
    colorscale:THEME.colormaps.occupancy,reversescale:THEME.occupancy_reverse,zmin:0,zmax:1,opacity:0.5,showscale:false,
    hovertemplate:'x=%{x:.1f} y=%{y:.1f}<br>P(occupied)=%{z:.2f}<extra></extra>'});
  for(const o of meta.objects){
    tr.push({x:o.x,y:o.y,type:'scatter',mode:'lines',fill:'toself',fillcolor:THEME.colors.object_fill_web,line:{color:THEME.colors.object_line,width:1},
      name:o.label,hoverinfo:'name'});
    const cx=o.x.slice(0,-1).reduce((a,b)=>a+b,0)/(o.x.length-1), cy=o.y.slice(0,-1).reduce((a,b)=>a+b,0)/(o.y.length-1);
    tr.push({x:[cx],y:[cy],type:'scatter',mode:'text',text:[o.label],textfont:{size:9,color:THEME.colors.white},hoverinfo:'skip'});
  }
  tr.push({x:[meta.start[0],meta.goal[0]],y:[meta.start[1],meta.goal[1]],type:'scatter',mode:'markers',text:['start','goal'],hoverinfo:'text',
    marker:{size:[9,15],color:THEME.colors.black,symbol:['circle','star']}});
  if(lid){
    tr.push({x:f.gnd.x,y:f.gnd.y,type:'scatter',mode:'markers',marker:{size:2,color:THEME.colors.ground_points},hoverinfo:'skip'});
    tr.push({x:f.obs.x,y:f.obs.y,type:'scatter',mode:'markers',marker:{size:4,color:THEME.colors.obstacle},
      hovertemplate:'obstacle point<br>x=%{x:.2f} y=%{y:.2f}<extra></extra>'});
  }
  if(f.L){
    const X=[],Y=[];
    for(let k=0;k<meta.H.length;k++){X.push(f.x,f.x+f.L[k]*Math.cos(meta.H[k]),null);Y.push(f.y,f.y+f.L[k]*Math.sin(meta.H[k]),null);}
    tr.push({x:X,y:Y,type:'scatter',mode:'lines',line:{color:THEME.colors.free_path_web,width:1},hoverinfo:'skip'});
  }
  const [rx,ry]=circ(f.x,f.y,Math.min(meta.rmax,60),90);
  tr.push({x:rx,y:ry,type:'scatter',mode:'lines',line:{color:THEME.colors.range_circle,width:1.2,dash:'dot'},hoverinfo:'skip'});
  const [ix,iy]=circ(f.x,f.y,meta.r_inf,48);
  tr.push({x:ix,y:iy,type:'scatter',mode:'lines',line:{color:THEME.colors.planner_clearance,width:1,dash:'dot'},name:'planner clearance r_inf',hoverinfo:'name'});
  tr.push({x:f.trail.x,y:f.trail.y,type:'scatter',mode:'lines',line:{color:THEME.colors.black,width:1.5},hoverinfo:'skip'});
  const [bx,by]=circ(f.x,f.y,R,28);
  tr.push({x:bx,y:by,type:'scatter',mode:'lines',fill:'toself',fillcolor:THEME.colors.black,line:{color:THEME.colors.black},hoverinfo:'skip'});
  tr.push({x:[f.x,f.x+0.9*Math.cos(f.th)],y:[f.y,f.y+0.9*Math.sin(f.th)],type:'scatter',mode:'lines',line:{color:THEME.colors.camera_fov,width:4},hoverinfo:'skip'});
  tr.push({x:[f.x,f.x+1.5*Math.cos(f.psi)],y:[f.y,f.y+1.5*Math.sin(f.psi)],type:'scatter',mode:'lines',line:{color:THEME.colors.goal_direction,width:3},hoverinfo:'skip'});
  const hf=meta.hfov/2, rc=6;
  tr.push({x:[f.x+rc*Math.cos(f.th+hf),f.x,f.x+rc*Math.cos(f.th-hf)],y:[f.y+rc*Math.sin(f.th+hf),f.y,f.y+rc*Math.sin(f.th-hf)],
    type:'scatter',mode:'lines',line:{color:THEME.colors.camera_fov,width:1.5},hoverinfo:'skip'});

  let xr=[-5,32], yr=[-9,9], rev='m'+run+mode;
  if(mode==='grid'){xr=[meta.extent[0],meta.extent[1]];yr=[meta.extent[2],meta.extent[3]];}
  if(mode==='follow'){xr=[f.x-9,f.x+9];yr=[f.y-6.5,f.y+6.5];rev='f'+f.i;}
  Plotly.react('map',tr,{font:{family:THEME.font_family},
    title:{text:'whole map (green = free, red = occupied, yellow = unknown)',font:{size:13}},
    xaxis:{range:xr,title:'x [m]'},yaxis:{range:yr,title:'y [m]',scaleanchor:'x',scaleratio:1},
    margin:{l:50,r:10,t:40,b:40},showlegend:false,uirevision:rev,plot_bgcolor:THEME.colors.white
  },{responsive:true,displaylogo:false,scrollZoom:true});
}
