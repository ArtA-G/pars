# Импорт библиотек, загрузка датасета
import pandas as pd
import nltk
import pymorphy2
from itertools import takewhile

df = pd.read_csv('test_data.csv')

# Создание датафрейма с репликами менеджеров
df_mng = df.query('role =="manager"')

# Создание датафрейма для заполнения результатами парсинга
df_pars = pd.DataFrame({'dlg_id': pd.Series(dtype='int'),
                   'greeting': pd.Series(dtype='str'),
                   'int_himself': pd.Series(dtype='str'),
                   'name': pd.Series(dtype='str'),
                   'company': pd.Series(dtype='str'),
                   'farewell': pd.Series(dtype='str'),
                   'chek': pd.Series(dtype='bool')})
df_pars['dlg_id'] = df_mng.dlg_id.unique()


greeting = {'здравствуйте', 'добрый', 'привет'}
farewell = {'хороший', 'хорошего', 'свидание',
            'добрый', 'доброго'}
himself = {'звать', 'это'}

morph = pymorphy2.MorphAnalyzer()
for index, dlg in enumerate(df_pars['dlg_id']):
    txt_grt = '' # Реплики с приветствиями
    company = [] # Название компаний
    
    for txt in df_mng.query('dlg_id == @dlg').iloc[:3, 3]: #  Выбор первых 3-х реплик в диалоге
        token = nltk.word_tokenize(txt)
        normal_form = []
        name = 'no_name'
              
        for word in token:
            p = morph.parse(word)[0]
            normal_form.append(p.normal_form)
            
            # находим имя человека и присваиваем его переменной 
            if 'Name' in p.tag and p.score >= 0.4:
                name = p.normal_form
                
        # Заполнение столбца с имененами менеджеров и
        # заполнение столбца с репликами, где менеджер представился
        if himself.intersection(normal_form) and name != 'no_name':
            df_pars.loc[index, 'name'] = name
            df_pars.loc[index, 'int_himself'] = txt        

        # Заполнение столбца с названиями компаний    
        if 'компания' in normal_form:           
            index_begin = normal_form.index('компания') + 1
            company = takewhile(lambda x: morph.parse(x)[0].tag.POS in ['NOUN', 'ADJF'], token[index_begin:])
            df_pars.loc[index, 'company'] = ' '.join(list(company))
          
        # Заполнение переменной репликами, где менеджер поздоровался
        # таких реплик может быть несколько, каждая реплика - предложение
        if greeting.intersection(normal_form):
            txt_grt = txt_grt + '. ' + txt
            
    # Заполнение столбца с репликами, где менеджер поздоровался
    if txt_grt[2:] != '':
        df_pars.loc[index, 'greeting'] = txt_grt[2:]

    #  Выбор последних 3-х реплик менеджера в диалоге 
    for txt in df_mng.query('dlg_id == @dlg').iloc[-3:, 3]:
        txt_frw = ''
        token = nltk.word_tokenize(txt)
        normal_form = []
        for word in token:
            p = morph.parse(word)[0]
            normal_form.append(p.normal_form)
            
        # Заполнение переменной репликами, где менеджер попрощался
        # таких реплик может быть несколько, каждая реплика - предложение
        if farewell.intersection(normal_form):
            txt_frw = txt_frw + '. ' + txt
    
    # Заполнение столбца с репликами, где менеджер попрощался
    if txt_frw[2:] != '':
        df_pars.loc[index, 'farewell'] = txt_frw[2:]
        
# Заполнение столбца проверки требования "поздороваться и попрощаться с клиентом"               
df_pars['chek'] =  df_pars['greeting'].notna() & df_pars['farewell'].notna()

# Запись результатов парсинга в файл
df_pars.to_csv('pars_result.csv', encoding='utf-8', index=False)