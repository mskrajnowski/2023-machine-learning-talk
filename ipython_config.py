c = get_config()

# disable Jedi, which seems to have problems with fastai import completion
c.IPCompleter.use_jedi = False
