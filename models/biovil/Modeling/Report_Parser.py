'''
Thanks Willie Boag
'''
import re
import os
import pickle
import pandas as pd

report_content_filepath = '/home/faith/datasets/mimic_512/training.csv'
reports_df = pd.read_csv(report_content_filepath)
dir = reports_df['image_path'].apply(lambda row: row.split('/'))
            
reports_df['pg'] = dir.apply(lambda row: row[1])
reports_df['pName'] = dir.apply(lambda row: row[2])
reports_df['sName'] = dir.apply(lambda row: row[3])

reports_df['key'] = reports_df.apply(lambda row: (row[4], row[5]), axis=1)
reports_dict = dict(zip(reports_df['key'], reports_df['report_content']))

class MIMIC_RE(object):
    def __init__(self):
        self._cached = {}

    def get(self, pattern, flags=0):
        key = hash((pattern, flags))
        if key not in self._cached:
            self._cached[key] = re.compile(pattern, flags=flags)

        return self._cached[key]

    def sub(self, pattern, repl, string, flags=0):
        return self.get(pattern, flags=flags).sub(repl, string)

    def rm(self, pattern, string, flags=0):
        return self.sub(pattern, '', string)

    def get_id(self, tag, flags=0):
        return self.get(r'\[\*\*.*{:s}.*?\*\*\]'.format(tag), flags=flags)

    def sub_id(self, tag, repl, string, flags=0):
        return self.get_id(tag).sub(repl, string)

def parse_report(row, rdpath, findings_only=False,impression_only=False, findings_impressions=False):
    mimic_re = MIMIC_RE()
    #path = os.path.join(rdpath, row['pGroup'], row['pName'], row['sName'], row['sName'] + '.txt')
    #with open(path, 'r') as f: #throws error for this, as txt files are not present
    #    report = f.read()

    ## New report text retriever logic
    try:
        row_key = ( row['pName'], row['sName'] ) 
        report = reports_dict[row_key] #should yield a report text
    except KeyError as e: #Corresponding report not found
        #err_txt = "Report Not Found: Patient ID {patient_id}, Study ID {study_id}"
        #print(err_txt.format(patient_id = row['pName'], study_id = row['sName']))
        return ""
    ## End

    report = report.lower()
    report = mimic_re.sub_id(r'(?:location|address|university|country|state|unit number)', 'LOC', report)
    report = mimic_re.sub_id(r'(?:year|month|day|date)', 'DATE', report)
    report = mimic_re.sub_id(r'(?:hospital)', 'HOSPITAL', report)
    report = mimic_re.sub_id(r'(?:identifier|serial number|medical record number|social security number|md number)',
                             'ID', report)
    report = mimic_re.sub_id(r'(?:age)', 'AGE', report)
    report = mimic_re.sub_id(r'(?:phone|pager number|contact info|provider number)', 'PHONE', report)
    report = mimic_re.sub_id(r'(?:name|initial|dictator|attending)', 'NAME', report)
    report = mimic_re.sub_id(r'(?:company)', 'COMPANY', report)
    report = mimic_re.sub_id(r'(?:clip number)', 'CLIP_NUM', report)

    report = mimic_re.sub((
        r'\[\*\*(?:'
        r'\d{4}'  # 1970
        r'|\d{0,2}[/-]\d{0,2}'  # 01-01
        r'|\d{0,2}[/-]\d{4}'  # 01-1970
        r'|\d{0,2}[/-]\d{0,2}[/-]\d{4}'  # 01-01-1970
        r'|\d{4}[/-]\d{0,2}[/-]\d{0,2}'  # 1970-01-01
        r')\*\*\]'
    ), 'DATE', report)
    report = mimic_re.sub(r'\[\*\*.*\*\*\]', 'OTHER', report)
    report = mimic_re.sub(r'(?:\d{1,2}:\d{2})', 'TIME', report)

    report = mimic_re.rm(r'_{2,}', report, flags=re.MULTILINE)
    report = mimic_re.rm(r'the study and the report were reviewed by the staff radiologist.', report)

    matches = list(mimic_re.get(r'(?P<title>[ \w()]+):').finditer(report)) #Remove carat and newline flag
    parsed_report = {}
    for (match, next_match) in zip(matches, matches[1:] + [None]):
        start = match.end()
        end = next_match and next_match.start()

        title = match.group('title')
        title = title.strip()

        paragraph = report[start:end]
        paragraph = mimic_re.sub(r'\s{2,}', ' ', paragraph)
        paragraph = paragraph.strip()

        parsed_report[title] = paragraph.replace('\n', '\\n')

    if findings_only:
        if 'findings' in parsed_report:
            return parsed_report['findings']
        else:
            return ""
    elif impression_only:
        if 'impression' in parsed_report:
            return parsed_report['impression']
        else:
            return ""
    elif findings_impressions:
        text = ""
        if 'findings' in parsed_report:
            text += "findings: " + parsed_report['findings']
        if 'impression' in parsed_report:
            text += "impression: " + parsed_report['impression']
        return text
    else:
        return parsed_report