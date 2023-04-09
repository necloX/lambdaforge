import numpy as np
from lambdaforge.lambda_term import Lambda

def parenthesis_de_bruijn(l:Lambda):
    word_abs_and_app = np.where(l.abstractions(), '(λ', '(')
    word_var = np.char.mod('%d', l.de_bruijn_indices)
    app_kernel = l.application_kernel
    closing_parenthesis = np.where(l.variables(), ')', '')
    h = l.height()
    closing_parenthesis = np.char.multiply(closing_parenthesis, (h-h[app_kernel]-1))
    closing_parenthesis[l.size-1] += '))'
    word_var = np.char.add(word_var, closing_parenthesis)
    word = np.where(l.variables(), word_var, word_abs_and_app)
    return ' '.join(word)  

def polish_de_bruijn(l:Lambda):
    word_abs_and_app = np.where(l.abstractions(), 'λ', '@')
    word_var = np.char.mod('%d', l.de_bruijn_indices)
    word = np.where(l.variables(), word_var, word_abs_and_app)
    return ' '.join(word)
def parenthesis_named(l:Lambda,abs_names,free_var_names):
    b = l.bindings()
    n = -np.min(b) + np.sum(l.abstractions())
    if(abs_names.size < np.sum(l.abstractions())): 
        print('not enough abstraction names')
        return 'fail'
    # Check if there are enough free variable names
    if(free_var_names.size < np.sum(l.abstractions())): 
        print('not enough free names')
        return 'fail'
    
    app_kernel = l.application_kernel
    closing_parenthesis = np.where( l.variables() , ')' , '' )
    h = l.height()
    closing_parenthesis = np.char.multiply(closing_parenthesis, (h-h[app_kernel]-1) )
    closing_parenthesis[l.size-1] += '))'  
    
    abstraction_names = abs_names[np.cumsum(l.abstractions())-1]
    abstraction_names_with_lambda = np.char.add(np.char.add('(λ',abstraction_names),'. ')
    var_names = np.char.add( abstraction_names[b],closing_parenthesis)
    f_var_names =np.char.add( free_var_names[ (l.de_bruijn_indices-l.height_abs()) * l.free_var()],closing_parenthesis)

    word = np.where(l.bounded_var(), var_names,
                    np.where(l.free_var(),f_var_names,
                             np.where(l.abstractions(),abstraction_names_with_lambda,
                                      '(')))
    return ''.join(word)

def polish_named(l:Lambda,abs_names,free_var_names,lambda_sym='λ',app_sym='@'):
    b = l.bindings()
    n = -np.min(b) + np.sum(l.abstractions())
    
    if(abs_names.size < np.sum(l.abstractions())): 
        print('not enough abstraction names')
        return 'fail'
    if(free_var_names.size < np.sum(l.abstractions())): 
        print('not enough free names')
        return 'fail'
    
    abstraction_names = abs_names[np.cumsum(l.abstractions())-1]
    abstraction_names_with_lambda = np.char.add(lambda_sym+' ',abstraction_names)
    var_names = abstraction_names[b * l.bounded_var()]
    free_var_names = free_var_names[ (l.de_bruijn_indices-l.height_abs()) * l.free_var()]
    word = np.where(l.bounded_var(), var_names,
                    np.where(l.free_var(),free_var_names,
                             np.where(l.abstractions(),abstraction_names_with_lambda,
                                      '@')))

    return ' '.join(word)     
