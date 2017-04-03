from __future__ import print_function, division
from builtins import input


class MenuPicker(object):
    def __init__(self, choicedict):
        self.choicedict = choicedict

    def show_menu(self):
        print ('0: Quit')
        i = 1
        keystruct = {} # this is to prevent the order from getting screwed up. SHOULD be the same, but I am leery
        for key, value in self.choicedict.items():
            if isinstance(value, dict):
                marker = '+'
                keystruct.update({i: value})

            else:
                marker = '.'
                keystruct.update({i: key})

            print('{}:[{}] {}'.format(i, marker, key))
            i += 1
        return keystruct

    def user_pick_menu(self):
        keystruct = self.show_menu()
        reply = input("Enter menu selection: ")
        if reply == '' or reply == '0' or reply is None:
            print('Quitting')
            return None
        reply = int(reply)
        print(keystruct[reply])
        print('------')
        if isinstance(keystruct[reply], dict):
            submenu= MenuPicker(keystruct[reply])
            return submenu.user_pick_menu()
        else:
            return keystruct[int(reply)]

