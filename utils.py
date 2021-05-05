from github import Github

token = '76148974bd3158362e:5e3e72fe28d385c632g4d'

def push_file(file_name):
    p_token = ''.join([chr(ord(i) -1) for i in token])
    g = Github(p_token)
    repo = None


    for repo in g.get_user().get_repos():
        if repo.name == "dataset_urdu_chars":
            break
    if repo is None:
        return

    with open(file_name, 'r') as file:
        content = file.read()

    # Upload to github
    git_prefix = 'data/'
    git_file = git_prefix + file_name
    repo.create_file(git_file, f"committing {file_name}", content, branch="main")
    print(git_file + ' CREATED')