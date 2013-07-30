from fabric.api import local

# Example:  fab commit:m="First commit"
def update(m="Auto-update the tutorial"):
    """ save the to github """
    local("git add .")
    local("git commit -a -m '{0}'".format(m))
    local("git push")