class HTMLReport:
    def __init__(self):
        self._header_elements = []
        self._elements = []

        self.add_header_element('link', params={'rel': '"stylesheet"',
                                                'href': '"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css'
                                                        '/bootstrap.min.css" '
                                                })
        self.add_header_element('style', data="body{ margin:0 100; background:whitesmoke; }")
        pass

    def add_header_element(self, element, data='', params={}):
        self._header_elements.append(self.get_html(element, data, params))

    def add_element(self, element, data='', params={}):
        self._elements.append(self.get_html(element, data, params))

    def add_graph(self, filename, pct_dims={33, 50}):
        self._elements.append(self.get_html('iframe', '', {'width': f'"{pct_dims[0]}%"',
                                                           'height': f'"{pct_dims[1]}%"',
                                                           'frameborder': '"0"',
                                                           'src': f'"{filename}"'}))

    @staticmethod
    def get_html(element, data='', params={}):
        params = ' ' + ' '.join([f'{p}={params[p]}' for p in params]) if len(params.keys()) > 0 else ''
        if element in ['hr', 'br', 'img', 'link']:
            return f"<{element}{params}>"
        else:
            return f"<{element}{params}>{data}</{element}>"

    def get_report(self):
        report = ['<html>', '\t<head>']
        for h in self._header_elements:
            report.append(f'\t\t{h}')
        report.extend(['\t</head>', '\t<body>'])
        for e in self._elements:
            report.append(f'\t\t{e}')
        report.extend(['\t</body>', '</html>'])
        return '\n'.join(report)


if __name__ == '__main__':
    a = HTMLReport()
    a.add_element('h1', 'Hello World!')
    a.add_element('p', 'This is a paragraph!')
    print(a.get_report())
