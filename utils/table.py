from bs4 import BeautifulSoup
from collections import defaultdict

def fix_rowspan(html_table):
    table = BeautifulSoup(html_table, features="html.parser")
    span_track = defaultdict(int)
    tag_to_remove = []
    for row_num, row in enumerate(table("tr")):
        for cell_num, cell in enumerate(row("td")):
            if 'rowspan' in cell.attrs and cell.attrs['rowspan'].isdigit():
                span_track[cell_num] = int(cell.attrs['rowspan'])-1
            elif cell_num in span_track and span_track[cell_num] > 0:
                span_track[cell_num] -= 1
                tag_to_remove.append(cell)
    for tag in tag_to_remove:
        tag.decompose()
    return str(table)