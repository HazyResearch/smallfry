
def get_git_hash():
   git_hash = None
   try:
       git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
       logging.info('Git hash {}'.format(git_hash))
   except FileNotFoundError:
       logging.info('Unable to get git hash.')
   return str(git_hash)