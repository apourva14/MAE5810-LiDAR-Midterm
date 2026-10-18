"""Build the mathematical project PDF from saved data and the report figures.

Run with Python containing numpy, matplotlib, reportlab, pypdf, and Pillow:
    python tools/build_model_notes.py
The saved scan is read only; this script does not run a new simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
SCAN_DATA = HERE / 'data/scan'
SCAN_MEDIA = HERE / 'outputs/scan'
DEFAULT_OUTPUT = ROOT / 'output/pdf/Midterm_Project_Status_and_Completion_Plan.pdf'
W, H = letter
MARGIN = 72
CW = W - 2 * MARGIN


def register_fonts():
    candidates = [Path('/System/Library/Fonts/Supplemental'), Path('C:/Windows/Fonts')]
    names = [('TNR', ['Times New Roman.ttf', 'times.ttf']),
             ('TNRB', ['Times New Roman Bold.ttf', 'timesbd.ttf']),
             ('TNRI', ['Times New Roman Italic.ttf', 'timesi.ttf'])]
    for name, variants in names:
        path = next((base / f for base in candidates for f in variants if (base / f).is_file()), None)
        if path is None:
            # The built-in Times family preserves the traditional layout elsewhere.
            return 'Times-Roman', 'Times-Bold', 'Times-Italic'
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('TNR', normal='TNR', bold='TNRB', italic='TNRI')
    return 'TNR', 'TNRB', 'TNRI'


class Equation(Flowable):
    def __init__(self, latex, number, work, placements, font):
        super().__init__()
        self.number, self.placements, self.font = number, placements, font
        self.path = work / f'equation-{number:02}.pdf'
        latex = re.sub(r'\\(mathbf|mathsf|mathcal|mathbb|mathrm)\s*([A-Za-z0-9])', r'\\\1{\2}', latex)
        latex = re.sub(r'\\le(?![A-Za-z])', r'\\leq', latex)
        latex = re.sub(r'\\ge(?![A-Za-z])', r'\\geq', latex)
        math_to_image('$' + latex.replace(r'\frac', r'\dfrac') + '$', str(self.path),
                      prop=FontProperties(family='STIXGeneral', size=11.5), format='pdf', color='black')
        page = PdfReader(self.path).pages[0]
        self.ew, self.eh = float(page.mediabox.width), float(page.mediabox.height)
        self.scale = min(1.0, (CW - 37) / self.ew)
        if self.scale < 0.80:
            raise ValueError(f'Equation {number} needs splitting into shorter expressions')
        self.width, self.height = CW, self.eh * self.scale + 12

    def draw(self):
        x, y = self.canv.absolutePosition((CW - self.ew * self.scale) / 2, 6)
        self.placements.append((self.canv.getPageNumber()-1, self.path, x, y, self.scale))
        self.canv.setFont(self.font, 10.5)
        self.canv.drawRightString(CW, 6 + self.eh * self.scale / 2 - 3, f'({self.number})')


class Workflow(Flowable):
    """Vector diagram; numbered stages connect in a single unambiguous direction."""
    def __init__(self, regular, bold):
        super().__init__()
        self.width, self.height = CW, 103
        self.regular, self.bold = regular, bold

    def draw(self):
        c = self.canv
        boxw, boxh, gap = 143, 33, 19.5
        items = [('1. Pose and beams', 'Sensor state and scan pattern'),
                 ('2. Visible surface', 'FOV, LOS, and intersections'),
                 ('3. Returned power', 'Material and atmosphere'),
                 ('6. Point cloud', 'XYZ and reported attributes'),
                 ('5. Range and noise', 'Echo range and uncertainty'),
                 ('4. Echo detection', 'Merging and thresholding')]
        c.setStrokeColor(colors.black)
        c.setLineWidth(.55)
        for i, (a,b) in enumerate(items):
            x = (i % 3) * (boxw + gap)
            y = 62 if i < 3 else 8
            c.rect(x,y,boxw,boxh,fill=0,stroke=1)
            c.setFont(self.bold,9.6); c.drawCentredString(x+boxw/2,y+21,a)
            c.setFont(self.regular,8.6); c.drawCentredString(x+boxw/2,y+8,b)
        def arrow(x1,y1,x2,y2):
            import math
            c.line(x1,y1,x2,y2)
            theta=math.atan2(y2-y1,x2-x1)
            for dt in (-.46,.46):
                c.line(x2,y2,x2-5*math.cos(theta+dt),y2-5*math.sin(theta+dt))
        for x in (boxw,2*boxw+gap): arrow(x,78.5,x+gap,78.5)
        arrow(CW-boxw/2,62,CW-boxw/2,41)
        for x in (boxw+gap,2*(boxw+gap)): arrow(x,24.5,x-gap,24.5)


def build(output: Path, figures: Path):
    regular,bold,italic=register_fonts()
    styles={
        'body':ParagraphStyle('body',fontName=regular,fontSize=11,leading=16.2,alignment=TA_JUSTIFY,spaceAfter=7),
        'small':ParagraphStyle('small',fontName=regular,fontSize=9.3,leading=12.3,spaceAfter=5),
        'title':ParagraphStyle('title',fontName=bold,fontSize=16,leading=20,alignment=TA_CENTER,spaceAfter=9),
        'subtitle':ParagraphStyle('subtitle',fontName=regular,fontSize=11,leading=15.5,alignment=TA_CENTER,spaceAfter=9),
        'section':ParagraphStyle('section',fontName=bold,fontSize=13,leading=17,spaceAfter=10,keepWithNext=True),
        'sub':ParagraphStyle('sub',fontName=bold,fontSize=11,leading=15,spaceBefore=6,spaceAfter=5,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName=regular,fontSize=9.5,leading=12.1),
        'head':ParagraphStyle('head',fontName=bold,fontSize=9.5,leading=12.1),
        'caption':ParagraphStyle('caption',fontName=regular,fontSize=9.3,leading=12.3,spaceAfter=8),
    }
    def p(t,s='body'): return Paragraph(t,styles[s])
    def sec(t): return p(t,'section')
    def sub(t): return p(t,'sub')
    def tab(head,rows,widths,pad=4):
        obj=Table([[p(v,'head') for v in head]]+[[p(v,'cell') for v in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        obj.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),
                                ('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.45,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black)]))
        return obj
    def fig(name,maxheight):
        path=figures/name
        if not path.exists(): raise FileNotFoundError(f'Missing report figure: {path}')
        with PILImage.open(path) as im: iw,ih=im.size
        scale=min(CW/iw,maxheight/ih)
        obj=Image(str(path),width=iw*scale,height=ih*scale)
        obj.hAlign='CENTER'
        return obj

    with (SCAN_DATA/'config_used.json').open() as f: cfg=json.load(f)
    sensor,pose=cfg['sensor'],cfg['pose']
    with (SCAN_DATA/'pointcloud.csv').open() as f:
        rdr=csv.reader(f);fields=next(rdr);count=sum(1 for _ in rdr)
    import numpy as np
    with np.load(SCAN_DATA/'pointcloud.npz',allow_pickle=False) as saved:
        names=saved['label_names'].tolist()
        ids=saved['label_id']
        object_count={name:int((ids==i).sum()) for i,name in enumerate(names)}
    placements=[]
    with tempfile.TemporaryDirectory(prefix='midterm-document-') as tmp:
        work=Path(tmp); equations=[]
        def eq(latex):
            obj=Equation(latex,len(equations)+1,work,placements,regular)
            equations.append(obj);return obj
        story=[]
        # 1: overview and navigation.
        story += [p('Mathematical Model of a 3D Pulsed LiDAR','title'),
                  p('MAE 5810 Robot Perception<br/>Existing model, analytical interpretation, and project status<br/>16 October 2026','subtitle'),
                  p('<b>Abstract.</b> This document organizes the current simulator into geometry, optical response, and measurement stages. '
                    'It derives the model dependencies required by the project and presents clearer views of the existing saved scan. '
                    'The figures are redrawn from the same 32,759 returns; no new simulation or physical validation is claimed.'),
                  sub('1. Purpose and reading guide'),
                  p('The assignment requires simulated measurements and field of view (FOV), explained dependencies on sensor pose, '
                    'targets, and environment, and graduate mathematical analysis [1]. The implementation is pulsed LiDAR [3]. '
                    'The supplied Heiden paper uses amplitude-modulated continuous-wave ranging [4], whose phase processing is not implemented.'),
                  Workflow(regular,bold),
                  p('Figure 1. Sensor-model pipeline. Read the top row left to right, then the bottom row right to left.','caption'),
                  tab(['Question','Section','Page'],[
                      ['How are rays positioned and which surfaces are visible?','2. Geometry and visibility','2'],
                      ['How do materials and atmosphere produce echoes?','3. Optical response and echo formation','3'],
                      ['How are detections and reported points generated?','4. Measurement and uncertainty','4'],
                      ['Why does the model capture the required dependencies?','5. Mathematical analysis','5'],
                      ['What scene and outputs already exist?','6-7. Configuration and visual results','6-7'],
                      ['What still needs evidence and submission work?','8. Evaluation and completion','8'],
                      ['How is the finite beam sampled?','Appendix A; references','9'],
                  ],[258,174,36]),
                  Spacer(1,9),
                  p('<b>Conventions.</b> World and sensor frames are right-handed; sensor x is forward, y left, and z up. '
                    'Equations use radians; plot axes use meters and degrees. Bold symbols denote vectors or matrices. '
                    'A reported point must be sampled, in sensing limits, unblocked, and strong enough to detect.','small'),PageBreak()]

        # 2: model geometry.
        story += [sec('2. Geometry, motion, and visibility'),sub('2.1 Scan pattern'),
                  p('Let M be azimuth samples per revolution, L the channel count, and f<sub>rot</sub> the scan rate. '
                    'The configured full sweep has M = 1,800 and L = 32. For L &gt; 1,'),
                  eq(r'\alpha_i=\alpha_{\min}+\frac{2\pi i}{M},\qquad \beta_j=\beta_{\min}+\frac{j(\beta_{\max}-\beta_{\min})}{L-1}.'),
                  eq(r'\mathbf{b}_{ij}=[\cos\beta_j\cos\alpha_i,\ \cos\beta_j\sin\alpha_i,\ \sin\beta_j]^{\mathsf T}.'),
                  p('For a full revolution i = 0,...,M - 1. A partial window samples the configured interval; L = 1 uses the upper elevation. '
                    'All channels at one azimuth share a firing time.'),sub('2.2 Pose during a sweep'),
                  eq(r't_i=\frac{\alpha_i-\alpha_{\min}}{2\pi f_{\mathrm{rot}}},\qquad \mathbf{o}_i=\mathbf{o}_0+\mathbf{v}t_i.'),
                  eq(r'R_i=R_z(\psi_0+\omega_zt_i)R_y(\vartheta_0)R_x(\varphi_0),\qquad \mathbf{d}_{ij}^{W}=R_i\mathbf{b}_{ij}.'),
                  p('The sensor origin o<sub>i</sub> translates at constant world velocity v. Yaw varies at rate omega<sub>z</sub>; '
                    'roll and pitch remain fixed. R<sub>i</sub> maps sensor directions to world directions.'),sub('2.3 FOV and opaque line of sight'),
                  eq(r'\mathbf{q}=R_i^{\mathsf T}(\mathbf{p}-\mathbf{o}_i),\quad r=\|\mathbf q\|,\quad a=\operatorname{atan2}(q_y,q_x),\quad e=\arcsin(q_z/r).'),
                  eq(r'\mathcal F_i=\{\mathbf p:r_{\min}\le r\le r_{\max},\ a\in\mathcal A,\ \beta_{\min}\le e\le\beta_{\max}\}.'),
                  p('A is the circular horizontal scan window. This is nominal coverage; discrete beams can miss small targets. '
                    'For each finite-width sub-ray d<sub>k</sub>, the nearest surface intersection is'),
                  eq(r'r_k=\min\{s>10^{-6}\ \mathrm m:\mathbf o_i+s\mathbf d_k\in\mathcal S\}.'),
                  p('A miss has infinite range. The nearest opaque hit is found before range gating, so a weak or too-near surface '
                    'still blocks farther objects. Plane, sphere, box, and capped cylinder geometry is supported. Beam sampling details are in Appendix A.','small'),PageBreak()]

        # 3: signal formation.
        story += [sec('3. Optical response and echo formation'),sub('3.1 Surface and atmospheric response'),
                  p('A nominal beam has K weighted sub-rays with weights w<sub>k</sub> summing to one. For a ray-facing unit normal '
                    'n<sub>k</sub>, incidence cosine mu<sub>k</sub>, diffuse weight rho<sub>d</sub>, specular weight rho<sub>s</sub>, and exponent m,'),
                  eq(r'\mu_k=\max(0,-\mathbf n_k^{\mathsf T}\mathbf d_k),\qquad f_k=\frac{\rho_{d,k}}{\pi}+\frac{\rho_{s,k}(m_k+2)}{2\pi}[\max(2\mu_k^2-1,0)]^{m_k}.'),
                  eq(r'A_r=\pi(D_r/2)^2,\quad C_L=P_tA_r\eta,\quad \kappa=\kappa_{\mathrm{air}}+a_{\mathrm{rain}}q_{\mathrm{rain}}^{b_{\mathrm{rain}}}.'),
                  eq(r'P_k=w_kC_L f_k\frac{\mu_k}{r_k^2}\exp(-2\kappa r_k),\qquad r_{\min}\le r_k\le r_{\max}.'),
                  p('P<sub>t</sub> is transmitter peak power, D<sub>r</sub> receiver aperture diameter, eta efficiency, and kappa extinction. '
                    'Invalid hits contribute zero power. Rain affects extinction only. The factor exp(-2 kappa r) represents two-way attenuation.'),
                  sub('3.2 Pulse gate and mixed echoes'),
                  p('The physical motivation is round-trip pulse propagation, but the code obtains ideal ranges from geometry rather than a waveform estimator:'),
                  eq(r'r=\frac{c\tau}{2},\qquad \Delta r_{\mathrm{gate}}=\frac{c\tau_{\mathrm{FWHM}}}{2},\qquad c=299\,792\,458\ \mathrm{m\,s^{-1}}.'),
                  p('Sub-ray hits are sorted by range. Adjacent hits are merged when their gap is at most the gate. '
                    'For the 5 ns pulse, the gate is about 0.749 m. For echo group G,'),
                  eq(r'P_G=\sum_{k\in G}P_k,\qquad \bar r_G=\frac{\sum_{k\in G}P_kr_k}{\max(P_G,10^{-300})}.'),
                  p('The ideal echo range is a power-weighted centroid. At an object edge it can fall between physical surfaces. '
                    'Grouping uses consecutive gaps, so a cluster can span more than one gate width.'),
                  sub('3.3 Interpretation limits'),
                  p('The surface law is an illustrative diffuse-plus-specular approximation. The retroreflector coefficient is an effective return gain. '
                    'The current model has no transmission, refraction, reflected secondary paths, rain-droplet returns, or fog backscatter. '
                    'The pulse gate approximates echo merging rather than full waveform processing.','small'),PageBreak()]

        # 4: distribution and data meaning.
        story += [sec('4. Detection, uncertainty, and output'),sub('4.1 Detection and return choice'),
                  p('For receiver-noise scale sigma<sub>P</sub> and threshold gamma, an existing echo candidate has'),
                  eq(r'\widetilde P_G=P_G+\nu_G,\quad \nu_G\sim\mathcal N(0,\sigma_P^2),\quad D_G=\mathbf1\{\widetilde P_G\ge\gamma\sigma_P\}.'),
                  eq(r'P_D(G)=\Phi\!\left(\frac{P_G}{\sigma_P}-\gamma\right),\qquad P_{\mathrm{any}}=1-\prod_G[1-P_D(G)].'),
                  p('Phi is the standard normal CDF; the product assumes independent echo-noise draws. First/last selects the nearest/farthest '
                    'detected group; strongest selects the largest noisy power. One nominal beam retains at most one return.'),sub('4.2 Conditional range uncertainty'),
                  eq(r'S=\frac{\widetilde P}{\sigma_P},\qquad \sigma_r^2(S)=\sigma_0^2+\frac{(ct_{\mathrm{rise}}/2)^2}{2\max(S,10^{-9})}.'),
                  eq(r'z=\bar r_G+\sigma_r(S)\xi,\qquad \xi\sim\mathcal N(0,1).'),
                  p('S is the exact amplitude/noise ratio called SNR in the code. The variance law is timing-inspired and assumed, '
                    'not a derived Cramer-Rao bound. Noise is applied after detection; conditioning on detection truncates the observed power distribution.'),
                  sub('4.3 Point coordinates and displayed attributes'),
                  eq(r'\widehat{\mathbf p}^{S}=z\mathbf b_{ij},\qquad \widehat{\mathbf p}^{W}=\mathbf o_i+z\mathbf d_{ij}^{W}.'),
                  eq(r'I=\operatorname{clip}\!\left(\frac{\widetilde P}{P_{\mathrm{sat}}},0,1\right),\qquad \rho_{\mathrm{app}}=\frac{\pi\widetilde Pz^2}{C_L}.'),
                  p('Both point coordinates use the central beam direction. World coordinates use the known firing-time pose; '
                    'they are not a pose estimate. Intensity is clipped display scaling. Apparent reflectivity retains incidence, atmospheric, '
                    'specular, noise, and mixing effects; it is not intrinsic calibrated reflectance.'),
                  p('Missing returns are absent from the point list and NaN in the range image. CSV/PLY store sensor coordinates; '
                    'NPZ additionally stores world coordinates, optical power, ideal echo range, angles, timestamps, and ground-truth object labels.','small'),PageBreak()]

        # 5: direct equation-to-implication analysis.
        story += [sec('5. Mathematical explanation of model behavior'),sub('5.1 Geometry and sensor state'),
                  p('For a plane through p<sub>0</sub> with unit normal n, a valid forward intersection has'),
                  eq(r'r_{\mathrm{plane}}=\frac{\mathbf n^{\mathsf T}(\mathbf p_0-\mathbf o)}{\mathbf n^{\mathsf T}\mathbf d},\qquad \mathbf n^{\mathsf T}\mathbf d\ne0.'),
                  p('For horizontal ground and a downward beam beta &lt; 0, mounting height h gives'),
                  eq(r'r=-\frac{h}{\sin\beta},\qquad \frac{\partial r}{\partial h}=-\frac{1}{\sin\beta},\qquad \frac{\partial r}{\partial\beta}=\frac{h\cos\beta}{\sin^2\beta}.'),
                  p('<b>Implication:</b> height and orientation change range; near-horizontal beams are highly sensitive. '
                    'These are pencil-beam reference predictions before finite-footprint mixing.'),sub('5.2 Target and environment'),
                  p('For a diffuse beam-filling target with fixed geometry, P = C exp(-2 kappa r)/r<super>2</super>, '
                    'where C = P<sub>t</sub>A<sub>r</sub>eta rho<sub>d</sub> cos(theta)/pi. Therefore'),
                  eq(r'\frac{\partial\ln P}{\partial r}=-\frac{2}{r}-2\kappa,\quad \frac{\partial\ln P}{\partial\kappa}=-2r,\quad \frac{\partial\ln P}{\partial\theta}=-\tan\theta.'),
                  p('<b>Implication:</b> greater distance, extinction, and grazing incidence weaken a diffuse return. '
                    'The angle derivative holds range fixed and excludes the specular branch.'),sub('5.3 Detection distance and uncertainty'),
                  eq(r'P(r_{50})=\gamma\sigma_P,\qquad r_{50}=\sqrt{\frac{C}{\gamma\sigma_P}}\quad(\kappa=0).'),
                  p('<b>Implication:</b> r<sub>50</sub> is a 50% single-echo detection distance, subject to FOV/LOS and range gates. '
                    'With extinction, solve the power equation numerically. For known direction and pose,'),
                  eq(r'\operatorname{Cov}(\widehat{\mathbf p}^{W}\mid\widetilde P,G)=\sigma_r^2(S)\mathbf d^W(\mathbf d^W)^{\mathsf T}.'),
                  p('<b>Implication:</b> the modeled range error lies along the beam. Angular, pose, calibration, and model errors are not included. '
                    'A localization or waveform estimator is not implemented.'),PageBreak()]

        # 6: scene, settings and proper plan view.
        story += [sec('6. Existing scene and configuration'),
                  p('The following figures redraw the existing saved scan. Figure 2 separates the faint ground returns from the named objects. '
                    'Reference outlines describe scene geometry; colored dots are measured returns.'),
                  fig('plan_view.png',305),
                  p('Figure 2. Plan view in world coordinates. The sensor marker and reference outlines locate the scene; '
                    'ground is subdued to make the six foreground objects readable. Colors match Figure 3 and the interactive view.','caption'),
                  p('Table 1. Main settings of the saved example (simulation inputs).','caption'),
                  tab(['Input','Setting'],[
                      ['Scan and angles','32 channels; 1,800 azimuths/revolution; 10 Hz; horizontal 360 deg; elevation -25 to +15 deg'],
                      ['Pose and range gate','Origin (0, 0, 1.8) m; zero motion; 0.5-200 m range'],
                      ['Transmitter and beam','75 W; full divergence 2 mrad; 12 sub-rays; 5 ns width; 1 ns rise'],
                      ['Receiver','25 mm aperture; efficiency 0.5; noise RMS 4 x 10<super>-9</super> W; threshold 10'],
                      ['Uncertainty and atmosphere','Range floor 0.01 m; air extinction 2 x 10<super>-4</super> m<super>-1</super>; rain 0 mm/h'],
                      ['Selected output',f'Strongest return; {count:,} saved points; {len(fields)} CSV fields'],
                  ],[145,CW-145],pad=4),
                  Spacer(1,8),
                  p('The scene contains asphalt ground, concrete and white walls, a black car-shaped box, an aluminium sphere, '
                    'a wooden pole, and a retroreflective box. These objects are stationary.','small'),PageBreak()]

        # 7: spatial and angular views at readable sizes.
        story += [sec('7. Existing visual results and how to read them'),
                  sub('7.1 Foreground object geometry'),fig('object_cloud.png',225),
                  p('Figure 3. Foreground returns in the world frame; ground returns are omitted only from this view. '
                    'The z axis is physical height. Missing portions can result from occlusion, unsampled directions, or weak signals.','caption'),
                  sub('7.2 Angular range image'),fig('range_image.png',170),
                  p('Figure 4. The same scan indexed by azimuth and elevation. Color represents measured range; gray means no return. '
                    'Gray pixels do not identify the cause of a missing return.','caption'),
                  p('<b>What the figures establish.</b> They show the saved spatial and angular measurements and make the scene understandable. '
                    'They do not establish calibrated physical accuracy or quantify the separate causes of missed returns. '
                    'Controlled comparisons and reference-error plots are still needed.'),PageBreak()]

        # 8: status, evidence and actionable scope.
        story += [sec('8. Evaluation and project completion'),sub('8.1 Current status against the assignment'),
                  tab(['Requirement','Current evidence','Remaining work'],[
                      ['Sensor measurements','Implemented model and saved scan','Check against independent reference cases'],
                      ['FOV and LOS','Angular/range gates and nearest-hit geometry','Show pose sweeps and boundary/occlusion cases'],
                      ['Required dependencies','Equations and configurable inputs','Compare sensor, target, and environment changes'],
                      ['MAE 5810 analysis','Derivations in Section 5','Connect predictions to measured simulation results'],
                      ['Report and individual media','This model document and revised views','Complete experimental analysis and final report'],
                  ],[113,165,190]),
                  sub('8.2 Quantitative evidence'),
                  p('For matched detected ranges, let e<sub>i</sub> = z<sub>i</sub> - r<sub>ref,i</sub>. Report range error separately from missed returns:'),
                  eq(r'\mathrm{bias}=\frac{1}{N_d}\sum_i e_i,\quad \mathrm{RMSE}=\sqrt{\frac{1}{N_d}\sum_i e_i^2},\quad \widehat P_D=\frac{N_{\mathrm{detected}}}{N_{\mathrm{eligible}}}.'),
                  p('Eligible opportunities are beams that geometrically strike an in-range target. Use analytical planes/spheres for geometry; '
                    'pose and occluder changes for FOV/LOS; distance, reflectivity and angle sweeps for targets; '
                    'extinction changes and repeated independent scans for environment and uncertainty.'),
                  p('Compare simple geometry, constant-noise, and fuller sensor models in identical scenes. Record runtime and beam count. '
                    'Agreement with implemented assumptions verifies consistency; real-sensor validation needs independent measured data.'),
                  sub('8.3 Final submission'),
                  p('Prepare the 5-10-page individual Word or LaTeX report with title page, abstract, introduction, methods, procedure, '
                    'results and mathematical analysis, conclusions, and references [1, 2]. Submit the PDF, editable source, '
                    'code/configuration, and distinct individual media. Hardware, a rover chassis, all five modeling approaches, and a video are optional.'),
                  p('Recommended order: verify the current implementation; run controlled comparisons; connect plots to the analysis; '
                    'assemble the final report. The current visual revision does not add new experimental evidence.','small'),PageBreak()]

        # 9: details do not interrupt the main measurement model.
        story += [sec('Appendix A. Finite-beam quadrature'),
                  p('The main model uses weighted sub-rays; this appendix records the sampler implemented in the code. '
                    'For full angular divergence delta, sigma<sub>b</sub> = delta/4. For K &gt; 1,'),
                  eq(r'a_k=2.5\sigma_b\sqrt{\frac{k+1/2}{K}},\quad \chi_k=k\pi(3-\sqrt{5}),\quad \mathbf q_k=a_k[\cos\chi_k,\sin\chi_k]^{\mathsf T},'),
                  eq(r'w_k=\frac{\exp[-a_k^2/(2\sigma_b^2)]}{\sum_{\ell=0}^{K-1}\exp[-a_\ell^2/(2\sigma_b^2)]},\qquad \sum_k w_k=1.'),
                  p('Let u and v<sub>b</sub> be an orthonormal basis transverse to nominal world direction d. The perturbed direction is'),
                  eq(r'\mathbf d_k=\frac{\mathbf d+q_{k1}\mathbf u+q_{k2}\mathbf v_b}{\|\mathbf d+q_{k1}\mathbf u+q_{k2}\mathbf v_b\|}.'),
                  p('The helper axis is world z unless the beam is almost vertical, when world x is used. '
                    'For K = 1 the offset is zero and w = 1. Gaussian weights are renormalized after truncation.'),
                  sub('A.1 Error interpretation at mixed edges'),
                  eq(r'\mathbb E[(z-r_{\mathrm{ref}})^2\mid\widetilde P,G]=(\bar r_G-r_{\mathrm{ref}})^2+\sigma_r^2(S).'),
                  p('The first term is displacement of the ideal echo centroid from a selected physical reference; the second is conditional random range variance. '
                    'Thus an edge point can remain off-surface when random noise is disabled. Reported apparent reflectivity likewise '
                    'retains angle, atmosphere, specular response, and mixing.'),
                  sub('References'),
                  p('[1] S. Ferrari, "Midterm Project," MAE 4810/5810 Robot Perception, supplied assignment, pp. 1-2.<br/>'
                    '[2] S. Ferrari, "Writing Guidelines," Robot Perception, Oct. 2, 2023, pp. 1-2.<br/>'
                    '[3] Project implementation: lidar_project/model/, configs/sensor.jsonc, and data/scan/ saved data. '
                    'Figures were redrawn from the saved NPZ archive without running the sensor simulation.<br/>'
                    '[4] E. Heiden, Z. Liu, R. K. Ramachandran, and G. S. Sukhatme, "Physics-based Simulation of Continuous-Wave LIDAR '
                    'for Localization, Calibration and Tracking," in <i>Proc. IEEE ICRA</i>, 2020.','small')]

        def page_style(c,doc):
            c.saveState();c.setFont(regular,9)
            if doc.page>1:c.drawString(MARGIN,H-42,'3D Pulsed LiDAR: Model, Analysis, and Existing Results')
            c.drawCentredString(W/2,35,str(doc.page));c.restoreState()
        base=work/'base.pdf'
        doc=BaseDocTemplate(str(base),pagesize=letter,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=MARGIN,bottomMargin=MARGIN)
        doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(MARGIN,MARGIN,CW,H-2*MARGIN,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page_style))
        doc.build(story)
        reader=PdfReader(base)
        for idx,path,x,y,scale in placements:
            reader.pages[idx].merge_transformed_page(PdfReader(path).pages[0],Transformation().scale(scale).translate(x,y),over=True)
        writer=PdfWriter()
        for page in reader.pages:writer.add_page(page).compress_content_streams()
        writer.add_metadata({'/Title':'3D Pulsed LiDAR: Mathematical Model, Analysis, and Existing Results',
                             '/Author':'MAE 5810 Robot Perception',
                             '/Subject':'Structured sensor model with clearer saved-data visualizations and project completion requirements'})
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_suffix('.revised.pdf')
        with temp.open('wb') as f:writer.write(f)
        temp.replace(output)
        print(json.dumps({'pdf':str(output),'pages':len(reader.pages),'equations':len(equations),'points':count}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--figures',type=Path,default=SCAN_MEDIA/'report_figures')
    parser.add_argument('--refresh-visuals',action='store_true',help='Redraw the saved scan, report figures, and interactive view; do not simulate')
    args=parser.parse_args()
    if args.refresh_visuals:
        import numpy as np
        from lidar_project.api import PointCloud, build_all
        from lidar_project.visualization.scan_figures import save_overview as plot_cloud
        from lidar_project.visualization.scan_viewer import plot_interactive
        from lidar_project.visualization.scan_figures import save_report_figures
        raw=json.loads((SCAN_DATA/'config_used.json').read_text())
        cfg,pose,scene,_,_=build_all({k:v for k,v in raw.items() if not k.startswith('_')})
        with np.load(SCAN_DATA/'pointcloud.npz',allow_pickle=False) as saved:
            data={key:saved[key] for key in PointCloud.__dataclass_fields__}
            data['label_names']=data['label_names'].tolist()
            data['n_beams']=int(data['n_beams'])
        cloud=PointCloud(**data)
        save_report_figures(cloud,scene,cfg,args.figures,sensor_position=pose.position)
        plot_cloud(cloud,str(SCAN_MEDIA/'pointcloud.png'),'3D LiDAR: saved scan',
                   scene=scene,cfg=cfg,sensor_position=pose.position)
        plot_interactive(cloud,scene,cfg,pose.position,str(SCAN_MEDIA/'pointcloud_interactive.html'),
                         '3D LiDAR: saved scan')
    matplotlib.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral','text.usetex':False})
    build(args.output.resolve(),args.figures.resolve())


if __name__=='__main__':main()
