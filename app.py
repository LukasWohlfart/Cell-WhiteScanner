import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
st.title('Whitescanner-Tool: Cell-Masking')

# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)
save_path = st.sidebar.text_input('Folder-Path')

all_pics = st.sidebar.file_uploader('', accept_multiple_files=True, type=("png", "jpg"))

thresh = st.sidebar.slider('White-Mask Threshold:',min_value = 0, max_value = 255,value = 130, step = 1)
#st.sidebar.write("White-Mask Threshold:", thresh)
st.write("White-Mask Threshold:", thresh)

def format_func(x):
    return x.name

selected = st.sidebar.selectbox('Select Picture', all_pics, format_func= format_func)

col1, col2 = st.beta_columns(2)

def to_excel_file(df, thresh):
    output = BytesIO()
    dict2 = {'threshold': ['threshold'], 'value': [float(thresh)]}
    df2 = pd.DataFrame(dict2) 

    column_settings = [{'header': column} for column in df.columns]
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Output', index = False)
    (max_row, max_col) = df.shape
    workbook  = writer.book
    worksheet = writer.sheets['Output']
    
    column_settings = [{'header': column} for column in df.columns]
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})
    df2.to_excel(writer, sheet_name= 'Output', startcol=0,startrow=max_row +3, header=None, index=False)
    for i, col in enumerate(df.columns):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col)) + 2
        # set the column length
        worksheet.set_column(i, i, column_len)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df, thresh):
    val = to_excel_file(df, thresh)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download xlsx file</a>' # decode b'abc' => abc

if selected is not None and len(save_path) > 5:
    img = Image.open(selected)
    img = np.array(img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    count = height * width
    ret,mask = cv2.threshold(img_gray,thresh,255,cv2.THRESH_BINARY)
    perc = np.round(cv2.countNonZero(mask)/count*100, decimals= 4)

    col1.header("Original")
    col1.image(img, use_column_width=True, channels="BGR")

    col2.header("Masked")
    col2.image(mask, use_column_width=True)

    st.write("Percent of White:", np.round(perc, decimals = 2), '%')

    percents = []
    for i in all_pics:
        if i is not None:
            img_i = Image.open(i)
            img_i = np.array(img_i.convert('RGB'))
            img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
            img_i_gray = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
            height_i, width_i = img_i_gray.shape
            count_i = height_i * width_i
            ret_i,mask_i = cv2.threshold(img_i_gray,thresh,255,cv2.THRESH_BINARY)
            perc_i = np.round(cv2.countNonZero(mask_i)/count_i*100, decimals= 4)
            percents.append(np.round(perc_i, decimals = 2))
            
    

    dict_data = {'Name': [el.name for el in all_pics],'Percent':percents}
    df_data = pd.DataFrame(dict_data)
    df_data = df_data.sort_values(by=['Name'])
    st.table(df_data.style.format({"Percent": "{:.2f}"}))
    #path_test = os.path.normpath(save_path)
    #dataname = path_test.split(os.sep)[-2]
    #foldername = path_test.split(os.sep)[-1]
    #if not os.path.exists(save_path+'/output_'+dataname+ '_'+foldername):
    #    os.makedirs(save_path+'/output_'+dataname+ '_'+foldername)
    col1, col2, col3, col4 = st.beta_columns(4)
    progress_bar = col2.progress(0)
    if col1.button('Save Images'):
        max_bar = len(all_pics)
        counter_bar = 0
        for i in all_pics:
            if i is not None:
                img_i = Image.open(i)
                img_i = np.array(img_i.convert('RGB'))
                img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
                img_i_gray = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
                height_i, width_i = img_i_gray.shape
                count_i = height_i * width_i
                ret_i,mask_i = cv2.threshold(img_i_gray,thresh,255,cv2.THRESH_BINARY)
                perc_i = np.round(cv2.countNonZero(mask_i)/count_i*100, decimals= 4)
                percents.append(np.round(perc_i, decimals = 2))

                

                vis = np.concatenate((img_i,cv2.cvtColor(mask_i,cv2.COLOR_GRAY2RGB)), axis=1)
                #cv2.imwrite(save_path+'/output_'+dataname+ '_'+foldername+'\\'+str(i.name)[:-4]+'_masked.jpg', vis)
                fig, ax = plt.subplots()
                ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                ax.text(1, 0, str(np.round(perc_i, decimals = 2))+' %', fontsize = 8, color = 'black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha = 0.7, pad = 2),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax.transAxes)
                ax.axis('off')
                # Add the patch to the Axes
                #fig.savefig(save_path+'/output_'+dataname+ '_'+foldername+'/'+str(i.name)[:-4]+'_masked.jpg', dpi=500, bbox_inches='tight', pad_inches=0)
                counter_bar += 1
                progress_bar.progress(counter_bar/max_bar)
    st.markdown(get_table_download_link(df_data, thresh), unsafe_allow_html=True)
    #if st.button('Download Excel'):
    #    to_excel_file(df_data, thresh, save_path+'/output_'+dataname+ '_'+foldername+'/'+'output.xlsx')
    
