"""Package the saved PDF content into an offline interactive HTML presentation.

This is a presentation exporter only: no sensor or navigation study is run.
Open the resulting index.html directly, or use python start.py to serve it.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def build(project=PROJECT,output=None):
    from plotly.offline import get_plotlyjs
    from lidar_project.visualization.report_data import load_weather_report_data
    from lidar_project.visualization.report_math import extract_equations
    project=Path(project).resolve()
    output=Path(output or project/'index.html').resolve()
    visual=project/'lidar_project/visualization'
    data=load_weather_report_data(project)
    data['equations']=extract_equations(project/'outputs/reports/Midterm_Project_Report.docx')
    data['provenance']['report_pdf']='outputs/reports/Midterm_Project_Report.pdf'
    data['provenance']['report_docx']='outputs/reports/Midterm_Project_Report.docx'
    template=(visual/'templates/report.html').read_text(encoding='utf-8')
    def image(path):
        return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    replacements={
        '__REPORT_DATA__':payload,
        '__REPORT_CSS__':(visual/'static/report.css').read_text(encoding='utf-8'),
        '__REPORT_JS__':(visual/'static/report.js').read_text(encoding='utf-8'),
        '__REPORT_3D_JS__':(visual/'static/report_3d.js').read_text(encoding='utf-8'),
        '__REPORT_CHARTS_JS__':(visual/'static/report_charts.js').read_text(encoding='utf-8'),
        '__PLOTLY_JS__':get_plotlyjs(),
        '__FOV_FALLBACK__':image(project/'outputs/weather_report/weather_fov_los.png'),
        '__MAP_FALLBACK__':image(project/'outputs/all_case_report/weather_maps.png'),
    }
    for token,value in replacements.items():
        template=template.replace(token,value)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template,encoding='utf-8')
    return {'html':str(output),'bytes':output.stat().st_size,'equations':len(data['equations']),
            'map_display_points':sum(len(row['points']) for row in data['mapping'].values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'index.html')
    arguments=parser.parse_args()
    print(json.dumps(build(output=arguments.output)))


if __name__=='__main__':
    main()
