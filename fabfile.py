"""
    REQUIREMENTS:
        - install pip with distribute (http://packages.python.org/distribute/)
        - sudo pip install Fabric

"""

from fabric.api import local

# Example:  fab run
def run():
    """ start the local app server """
    local("python demo.py")

# Example:  fab commit:m="First commit"
def commit(m="Fab-update the app"):
    """ save the to github """
    local("git add .")
    local("git commit -a -m '{0}'".format(m))
    local("git push")
