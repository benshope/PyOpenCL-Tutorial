from fabric.api import local

# Example:  fab run
def run():
    """ run the application """
    local("python main.py")

# Example:  fab commit:m="First commit"
def update(m="Fab-update the app"):
    """ save the to github """
    local("git add .")
    local("git commit -a -m '{0}'".format(m))
    local("git push")