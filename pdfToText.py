import pdftotext

def convert(pdf_fname, output_fname):
    with open(pdf_fname, "rb") as f1:
        pdf = pdftotext.PDF(f1)
    with open(output_fname, "w+") as f2:
        f2.write("\n\n".join(pdf))