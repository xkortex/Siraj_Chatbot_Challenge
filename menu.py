class Choice(object):
    """
    Represents an item in a command line menu
    """
    def __init__(self, key_or_tup, name=None, callback=None, **kwargs):
        """
        Generate a new menu item
        :param key_or_tup: str, tup, or Choice: Menu key, or list of arguments, or Choice object to clone
        :param name: Name of the menu item - this is printed alongside key
        :param callback: Function to activate when option is selected
        :param kwargs: Keyword arguments to feed to callback
        """
        self.kwargs = kwargs
        if isinstance(key_or_tup, Choice):
            key = key_or_tup.key
            name = key_or_tup.name
            callback = key_or_tup.callback
        elif isinstance(key_or_tup, (tuple, list)):
            if len(key_or_tup) == 3:
                key, name, callback = key_or_tup
            elif len(key_or_tup) == 2:
                key, name = key_or_tup
            elif len(key_or_tup) == 1:
                key = key_or_tup[0]
            else:
                raise ValueError('Invalid menu choice specification list')
        elif isinstance(key_or_tup, str):
            key = key_or_tup
        else:
            raise ValueError('Invalid menu choice specification list')
        self.key = key
        self.name = name
        self.callback = callback

    def __str__(self):
        return '{: >2}: {}'.format(self.key, self.name)

    def __call__(self, *args, **kwargs):
        if self.callback is not None:
            return self.callback(**self.kwargs)



class Quit(Choice):
    def __init__(self):
        super().__init__('q', 'Quit')

    def __call__(self):
        print('Quitting')


class Menu(Choice):
    """
        A special choice object, which when called, returns a new context menu

    """

    def __init__(self, key, name=None, choices=None, loop_on_invalid=False):
        """
        Create a Menu object, which when called, returns a new context menu
        :param key: Key option to call this menu
        :param name: Name of menu to disploy
        :param choices: List of Choice objects. Quit is automatically prepended
        :param loop_on_invalid: Display the menu again if selection was invalid. Otherwise, quit on invalid
        """
        self.loop_on_invalid = loop_on_invalid
        self.quit = Quit()
        choices = [self.quit] if choices is None else [self.quit] + choices
        self.choices = choices
        self.name = name
        super().__init__(key_or_tup=key, name=name, callback=self)

    def get_item(self, key):
        """
        Look up the key associated with a Choice, if it exists
        :param key:
        :return: Choice object associated with key
        """
        lookup = {choice.key: choice for choice in self.choices}
        if key == '' and self.loop_on_invalid:
            return Choice('', 'Nothing selected', lambda: print('Nothing selected'))
        elif key in lookup:
            return lookup[key]
        else:
            print('Invalid entry!')
            if self.loop_on_invalid:
                print('Enter menu selection: ')
            else:
                self.quit()

    def show_menu(self):
        """
        Show the CLI selection menu
        :return:
        """
        print('calling show_menu' + self.name)
        print(self.name)
        for choice in self.choices:
            print(choice)

    def add(self, choice):
        """
        Add Choice objects to the menu list
        :param choice:
        :return:
        """
        self.choices.append(Choice(choice))

    def user_pick_menu(self):
        """
        Get a reply from the user
        :return:
        """
        reply = input("Enter menu selection: ")
        return reply

    def __call__(self, *args, **kwargs):
        self.show_menu()
        r = self.user_pick_menu()
        choice = self.get_item(r)
        if choice is not None:
            return choice()


"""
Some examples:

a = Choice('a', 'this is a choice', return_foo)
b = Choice('b', 'this is another choice', lambda: print('choice b'))
c = Choice('c', 'this is 3rd choice', return_foo)
d = Choice('d', 'this is 4th choice', lambda: print('choice d'))
e = Choice('e', 'print 666', meprint, foo='666')

m = Menu('m', 'Menu 1', [a, b, e])
m2 = Menu('5', 'Menu 2', [c, d, e, m], True)

"""