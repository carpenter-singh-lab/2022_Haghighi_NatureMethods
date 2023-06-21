import pandas as pd
import openpyxl as pxl
import os

# ------------------------------------------------------


# Save the input dataframe to the specified sheet name of filename file
def saveAsNewSheetToExistingFile(filename, newDF, newSheetName):


    
    if os.path.exists(filename):
        excel_book = pxl.load_workbook(filename)

        if newSheetName in excel_book.sheetnames:
            del excel_book[newSheetName]

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            writer.book = excel_book

            writer.sheets = {
                worksheet.title: worksheet
                for worksheet in excel_book.worksheets
                if newSheetName not in worksheet
            }
            newDF.to_excel(writer, newSheetName)
            writer.save()
    else:
        newDF.to_excel(filename, newSheetName)

    print(newSheetName, " saved!")
    return


# ------------------------------------------------------


# saveDF_to_CSV_GZ_no_timestamp
def saveDF_to_CSV_GZ_no_timestamp(df, filename):
    from gzip import GzipFile
    from io import TextIOWrapper

    with TextIOWrapper(GzipFile(filename, "w", mtime=0), encoding="utf-8") as fd:
        df.to_csv(fd, index=False, compression="gzip")

    return
