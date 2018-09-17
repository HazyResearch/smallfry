
def get_git_hash():
   git_hash = None
   try:
       git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
       logging.info('Git hash {}'.format(git_hash))
   except FileNotFoundError:
       logging.info('Unable to get git hash.')
   return str(git_hash)


def codes_2_vec(codes, codebook, m ,k ,v,d):
    codes = codes.reshape(int(len(codes)/m),m)
    dcc_mat = np.zeros([v,d])
    for i in range(v):
        for j in range(m):
            dcc_mat[i,:] += codebook[j*k+codes[i,j],:]
    return dcc_mat