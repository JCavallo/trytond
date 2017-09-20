# This file is part of Tryton.  The COPYRIGHT file at the top level of
# this repository contains the full copyright notices and license terms.
from __future__ import division

from functools import wraps

from sql.operators import NotIn

import logging
from trytond.model import ModelView, ModelSQL, fields, Unique
from trytond.modules import create_graph, get_module_list, get_module_info
from trytond.wizard import Wizard, StateView, Button, StateTransition, \
    StateAction
from trytond.report import Report
from trytond import backend
from trytond.pool import Pool
from trytond.transaction import Transaction
from trytond.pyson import Eval, If
from trytond.rpc import RPC
from trytond.iwc import broadcast_init_pool

__all__ = [
    'Module', 'ModuleDependency', 'ModuleConfigWizardItem',
    'ModuleConfigWizardFirst', 'ModuleConfigWizardOther',
    'ModuleConfigWizardDone', 'ModuleConfigWizard',
    'ModuleActivateUpgradeStart', 'ModuleActivateUpgradeDone',
    'ModuleActivateUpgrade', 'ModuleConfig', 'PrintModuleGraph', 'ModuleGraph',
    'PrintModuleGraphParameters',
    ]

HAS_PYDOT = False
try:
    import pydot
    HAS_PYDOT = True
except ImportError:
    logging.getLogger('ir').warning(
            'Unable to import pydot, graph representation will be disabled')


def filter_state(state):
    def filter(func):
        @wraps(func)
        def wrapper(cls, modules):
            modules = [m for m in modules if m.state == state]
            return func(cls, modules)
        return wrapper
    return filter


class Module(ModelSQL, ModelView):
    "Module"
    __name__ = "ir.module"
    name = fields.Char("Name", readonly=True, required=True)
    version = fields.Function(fields.Char('Version'), 'get_version')
    dependencies = fields.One2Many('ir.module.dependency',
        'module', 'Dependencies', readonly=True)
    parents = fields.Function(fields.One2Many('ir.module', None, 'Parents'),
        'get_parents')
    childs = fields.Function(fields.One2Many('ir.module', None, 'Childs'),
        'get_childs')
    state = fields.Selection([
        ('not activated', 'Not Activated'),
        ('activated', 'Activated'),
        ('to upgrade', 'To be upgraded'),
        ('to remove', 'To be removed'),
        ('to activate', 'To be activated'),
        ], string='State', readonly=True)

    @classmethod
    def __setup__(cls):
        super(Module, cls).__setup__()
        table = cls.__table__()
        cls._sql_constraints = [
            ('name_uniq', Unique(table, table.name),
                'The name of the module must be unique!'),
        ]
        cls._order.insert(0, ('name', 'ASC'))
        cls.__rpc__.update({
                'on_write': RPC(instantiate=0),
                })
        cls._error_messages.update({
            'delete_state': ('You can not remove a module that is activated '
                    'or will be activated'),
            'missing_dep': 'Missing dependencies %s for module "%s"',
            'deactivate_dep': ('Some activated modules depend on the ones '
                    'you are trying to deactivate:'),
            })
        cls._buttons.update({
                'activate': {
                    'invisible': Eval('state') != 'not activated',
                    },
                'activate_cancel': {
                    'invisible': Eval('state') != 'to activate',
                    },
                'deactivate': {
                    'invisible': Eval('state') != 'activated',
                    },
                'deactivate_cancel': {
                    'invisible': Eval('state') != 'to remove',
                    },
                'upgrade': {
                    'invisible': Eval('state') != 'activated',
                    },
                'upgrade_cancel': {
                    'invisible': Eval('state') != 'to upgrade',
                    },
                })

    @classmethod
    def __register__(cls, module_name):
        TableHandler = backend.get('TableHandler')

        # Migration from 3.6: remove double module
        old_table = 'ir_module_module'
        if TableHandler.table_exist(old_table):
            TableHandler.table_rename(old_table, cls._table)

        super(Module, cls).__register__(module_name)

        # Migration from 4.0: rename installed to activated
        sql_table = cls.__table__()
        cursor = Transaction().connection.cursor()
        cursor.execute(*sql_table.update(
                [sql_table.state], ['activated'],
                where=sql_table.state == 'installed'))
        cursor.execute(*sql_table.update(
                [sql_table.state], ['not activated'],
                where=sql_table.state == 'uninstalled'))

    @staticmethod
    def default_state():
        return 'not activated'

    def get_version(self, name):
        return get_module_info(self.name).get('version', '')

    @classmethod
    def get_parents(cls, modules, name):
        parent_names = list(set(d.name for m in modules
                    for d in m.dependencies))
        parents = cls.search([
                ('name', 'in', parent_names),
                ])
        name2id = dict((m.name, m.id) for m in parents)
        return dict((m.id, [name2id[d.name] for d in m.dependencies])
            for m in modules)

    @classmethod
    def get_childs(cls, modules, name):
        child_ids = dict((m.id, []) for m in modules)
        name2id = dict((m.name, m.id) for m in modules)
        childs = cls.search([
                ('dependencies.name', 'in', name2id.keys()),
                ])
        for child in childs:
            for dep in child.dependencies:
                if dep.name in name2id:
                    child_ids[name2id[dep.name]].append(child.id)
        return child_ids

    @classmethod
    def view_attributes(cls):
        return [('/tree', 'colors',
                If(Eval('state').in_(['to upgrade', 'to install']),
                    'blue',
                    If(Eval('state') == 'uninstalled',
                        'grey',
                        'black')))]

    @classmethod
    def delete(cls, records):
        for module in records:
            if module.state in (
                    'activated',
                    'to upgrade',
                    'to remove',
                    'to activate',
                    ):
                cls.raise_user_error('delete_state')
        return super(Module, cls).delete(records)

    @classmethod
    def on_write(cls, modules):
        dependencies = set()

        def get_parents(module):
            parents = set(p.id for p in module.parents)
            for p in module.parents:
                parents.update(get_parents(p))
            return parents

        def get_childs(module):
            childs = set(c.id for c in module.childs)
            for c in module.childs:
                childs.update(get_childs(c))
            return childs

        for module in modules:
            dependencies.update(get_parents(module))
            dependencies.update(get_childs(module))
        return list(dependencies)

    @classmethod
    @ModelView.button
    @filter_state('not activated')
    def activate(cls, modules):
        modules_activated = set(modules)
        graph, packages, later = create_graph(get_module_list())

        def get_parents(module):
            parents = set(p for p in module.parents)
            for p in module.parents:
                parents.update(get_parents(p))
            return parents

        for module in modules:
            if module.name not in graph:
                missings = []
                for package, deps, xdep, info in packages:
                    if package == module.name:
                        missings = [x for x in deps if x not in graph]
                cls.raise_user_error('missing_dep', (missings, module.name))

            modules_activated.update((m for m in get_parents(module)
                    if m.state == 'not activated'))
        cls.write(list(modules_activated), {
                'state': 'to activate',
                })

    @classmethod
    @ModelView.button
    @filter_state('activated')
    def upgrade(cls, modules):
        modules_activated = set(modules)
        graph, packages, later = create_graph(get_module_list())

        def get_childs(module):
            childs = set(c for c in module.childs)
            for c in module.childs:
                childs.update(get_childs(c))
            return childs

        for module in modules:
            if module.name not in graph:
                missings = []
                for package, deps, xdep, info in packages:
                    if package == module.name:
                        missings = [x for x in deps if x not in graph]
                cls.raise_user_error('missing_dep', (missings, module.name))

            modules_activated.update((m for m in get_childs(module)
                    if m.state == 'activated'))
        cls.write(list(modules_activated), {
                'state': 'to upgrade',
                })

    @classmethod
    @ModelView.button
    @filter_state('to activate')
    def activate_cancel(cls, modules):
        cls.write(modules, {
                'state': 'not activated',
                })

    @classmethod
    @ModelView.button
    @filter_state('activated')
    def deactivate(cls, modules):
        pool = Pool()
        Module = pool.get('ir.module')
        Dependency = pool.get('ir.module.dependency')
        module_table = Module.__table__()
        dep_table = Dependency.__table__()
        cursor = Transaction().connection.cursor()
        for module in modules:
            cursor.execute(*dep_table.join(module_table,
                    condition=(dep_table.module == module_table.id)
                    ).select(module_table.state, module_table.name,
                    where=(dep_table.name == module.name)
                    & NotIn(
                        module_table.state, ['not activated', 'to remove'])))
            res = cursor.fetchall()
            if res:
                cls.raise_user_error('deactivate_dep',
                        error_description='\n'.join(
                            '\t%s: %s' % (x[0], x[1]) for x in res))
        cls.write(modules, {'state': 'to remove'})

    @classmethod
    @ModelView.button
    @filter_state('to remove')
    def deactivate_cancel(cls, modules):
        cls.write(modules, {'state': 'not activated'})

    @classmethod
    @ModelView.button
    @filter_state('to upgrade')
    def upgrade_cancel(cls, modules):
        cls.write(modules, {'state': 'activated'})

    @classmethod
    def update_list(cls):
        'Update the list of available packages'
        count = 0
        module_names = get_module_list()

        modules = cls.search([])
        name2module = dict((m.name, m) for m in modules)

        # iterate through activated modules and mark them as being so
        for name in module_names:
            if name in name2module:
                module = name2module[name]
                tryton = get_module_info(name)
                cls._update_dependencies(module, tryton.get('depends', []))
                continue

            tryton = get_module_info(name)
            if not tryton:
                continue
            module, = cls.create([{
                        'name': name,
                        'state': 'not activated',
                        }])
            count += 1
            cls._update_dependencies(module, tryton.get('depends', []))
        return count

    @classmethod
    def _update_dependencies(cls, module, depends=None):
        pool = Pool()
        Dependency = pool.get('ir.module.dependency')
        Dependency.delete([x for x in module.dependencies
            if x.name not in depends])
        if depends is None:
            depends = []
        # Restart Browse Cache for deleted dependencies
        module = cls(module.id)
        dependency_names = [x.name for x in module.dependencies]
        to_create = []
        for depend in depends:
            if depend not in dependency_names:
                to_create.append({
                        'module': module.id,
                        'name': depend,
                        })
        if to_create:
            Dependency.create(to_create)


class ModuleDependency(ModelSQL, ModelView):
    "Module dependency"
    __name__ = "ir.module.dependency"
    name = fields.Char('Name')
    module = fields.Many2One('ir.module', 'Module', select=True,
       ondelete='CASCADE', required=True)
    state = fields.Function(fields.Selection([
                ('not activated', 'Not Activated'),
                ('activated', 'Activated'),
                ('to upgrade', 'To be upgraded'),
                ('to remove', 'To be removed'),
                ('to activate', 'To be activated'),
                ('unknown', 'Unknown'),
                ], 'State', readonly=True), 'get_state')

    @classmethod
    def __setup__(cls):
        super(ModuleDependency, cls).__setup__()
        table = cls.__table__()
        cls._sql_constraints += [
            ('name_module_uniq', Unique(table, table.name, table.module),
                'Dependency must be unique by module!'),
        ]

    @classmethod
    def __register__(cls, module_name):
        TableHandler = backend.get('TableHandler')

        # Migration from 3.6: remove double module
        old_table = 'ir_module_module_dependency'
        if TableHandler.table_exist(old_table):
            TableHandler.table_rename(old_table, cls._table)

        super(ModuleDependency, cls).__register__(module_name)

    def get_state(self, name):
        pool = Pool()
        Module = pool.get('ir.module')
        dependencies = Module.search([
                ('name', '=', self.name),
                ])
        if dependencies:
            return dependencies[0].state
        else:
            return 'unknown'


class ModuleConfigWizardItem(ModelSQL, ModelView):
    "Config wizard to run after activating a module"
    __name__ = 'ir.module.config_wizard.item'
    action = fields.Many2One('ir.action', 'Action', required=True,
        readonly=True)
    sequence = fields.Integer('Sequence', required=True)
    state = fields.Selection([
        ('open', 'Open'),
        ('done', 'Done'),
        ], string='State', required=True, select=True)

    @classmethod
    def __setup__(cls):
        super(ModuleConfigWizardItem, cls).__setup__()
        cls._order.insert(0, ('sequence', 'ASC'))

    @classmethod
    def __register__(cls, module_name):
        TableHandler = backend.get('TableHandler')
        cursor = Transaction().connection.cursor()
        pool = Pool()
        ModelData = pool.get('ir.model.data')
        model_data = ModelData.__table__()

        # Migration from 3.6: remove double module
        old_table = 'ir_module_module_config_wizard_item'
        if TableHandler.table_exist(old_table):
            TableHandler.table_rename(old_table, cls._table)
        cursor.execute(*model_data.update(
                columns=[model_data.model],
                values=[cls.__name__],
                where=(model_data.model ==
                    'ir.module.module.config_wizard.item')))

        table = TableHandler(cls, module_name)

        # Migrate from 2.2 remove name
        table.drop_column('name')

        super(ModuleConfigWizardItem, cls).__register__(module_name)

    @staticmethod
    def default_state():
        return 'open'

    @staticmethod
    def default_sequence():
        return 10


class ModuleConfigWizardFirst(ModelView):
    'Module Config Wizard First'
    __name__ = 'ir.module.config_wizard.first'


class ModuleConfigWizardOther(ModelView):
    'Module Config Wizard Other'
    __name__ = 'ir.module.config_wizard.other'

    percentage = fields.Float('Percentage', readonly=True)

    @staticmethod
    def default_percentage():
        pool = Pool()
        Item = pool.get('ir.module.config_wizard.item')
        done = Item.search([
            ('state', '=', 'done'),
            ], count=True)
        all = Item.search([], count=True)
        return done / all


class ModuleConfigWizardDone(ModelView):
    'Module Config Wizard Done'
    __name__ = 'ir.module.config_wizard.done'


class ModuleConfigWizard(Wizard):
    'Run config wizards'
    __name__ = 'ir.module.config_wizard'

    class ConfigStateAction(StateAction):

        def __init__(self):
            StateAction.__init__(self, None)

        def get_action(self):
            pool = Pool()
            Item = pool.get('ir.module.config_wizard.item')
            Action = pool.get('ir.action')
            items = Item.search([
                ('state', '=', 'open'),
                ], limit=1)
            if items:
                item = items[0]
                Item.write([item], {
                        'state': 'done',
                        })
                return Action.get_action_values(item.action.type,
                    [item.action.id])[0]

    start = StateTransition()
    first = StateView('ir.module.config_wizard.first',
        'ir.module_config_wizard_first_view_form', [
            Button('Cancel', 'end', 'tryton-cancel'),
            Button('OK', 'action', 'tryton-ok', default=True),
            ])
    other = StateView('ir.module.config_wizard.other',
        'ir.module_config_wizard_other_view_form', [
            Button('Cancel', 'end', 'tryton-cancel'),
            Button('Next', 'action', 'tryton-go-next', default=True),
            ])
    action = ConfigStateAction()
    done = StateView('ir.module.config_wizard.done',
        'ir.module_config_wizard_done_view_form', [
            Button('OK', 'end', 'tryton-ok', default=True),
            ])

    def transition_start(self):
        res = self.transition_action()
        if res == 'other':
            return 'first'
        return res

    def transition_action(self):
        pool = Pool()
        Item = pool.get('ir.module.config_wizard.item')
        items = Item.search([
                ('state', '=', 'open'),
                ])
        if items:
            return 'other'
        return 'done'

    def end(self):
        return 'reload menu'


class ModuleActivateUpgradeStart(ModelView):
    'Module Activate Upgrade Start'
    __name__ = 'ir.module.activate_upgrade.start'
    module_info = fields.Text('Modules to update', readonly=True)


class ModuleActivateUpgradeDone(ModelView):
    'Module Activate Upgrade Done'
    __name__ = 'ir.module.activate_upgrade.done'


class ModuleActivateUpgrade(Wizard):
    "Activate / Upgrade modules"
    __name__ = 'ir.module.activate_upgrade'

    start = StateView('ir.module.activate_upgrade.start',
        'ir.module_activate_upgrade_start_view_form', [
            Button('Cancel', 'end', 'tryton-cancel'),
            Button('Start Upgrade', 'upgrade', 'tryton-ok', default=True),
            ])
    upgrade = StateTransition()
    done = StateView('ir.module.activate_upgrade.done',
        'ir.module_activate_upgrade_done_view_form', [
            Button('OK', 'config', 'tryton-ok', default=True),
            ])
    config = StateAction('ir.act_module_config_wizard')

    @classmethod
    def check_access(cls):
        # Use new transaction to prevent lock when activating modules
        with Transaction().new_transaction():
            super(ModuleActivateUpgrade, cls).check_access()

    @staticmethod
    def default_start(fields):
        pool = Pool()
        Module = pool.get('ir.module')
        modules = Module.search([
                ('state', 'in', ['to upgrade', 'to remove', 'to activate']),
                ])
        return {
            'module_info': '\n'.join(x.name + ': ' + x.state
                for x in modules),
            }

    def __init__(self, session_id):
        pass

    def _save(self):
        pass

    def transition_upgrade(self):
        pool = Pool()
        Module = pool.get('ir.module')
        Lang = pool.get('ir.lang')
        with Transaction().new_transaction():
            modules = Module.search([
                ('state', 'in', ['to upgrade', 'to remove', 'to activate']),
                ])
            update = [m.name for m in modules]
            langs = Lang.search([
                ('translatable', '=', True),
                ])
            lang = [x.code for x in langs]
        if update:
            pool.init(update=update, lang=lang)
            broadcast_init_pool()
        return 'done'


class ModuleConfig(Wizard):
    'Configure Modules'
    __name__ = 'ir.module.config'

    start = StateAction('ir.act_module_form')

    @staticmethod
    def transition_start():
        return 'end'


class PrintModuleGraphParameters(ModelView):
    'Print Module Graph Parameters'

    __name__ = 'ir.module.print_module_graph.parameters'

    trim_links = fields.Boolean('Group dependencies with parents')
    only_installed = fields.Boolean('Only installed modules')


class PrintModuleGraph(Wizard):
    __name__ = 'ir.module.print_module_graph'

    start_state = 'check_pydot'
    check_pydot = StateTransition()
    parameters = StateView('ir.module.print_module_graph.parameters',
        'ir.print_module_graph_parameters_form', [
            Button('Cancel', 'end', 'tryton-cancel'),
            Button('Generate', 'graph', 'tryton-go-next')])
    graph = StateAction('ir.report_module_graph')

    @classmethod
    def __setup__(cls):
        super(PrintModuleGraph, cls).__setup__()
        cls._error_messages.update({
            'no_pydot':
            'Pydot cannot be found. This functionnality is disabled',
        })

    def do_graph(self, action):
        return action, {
            'trim_links': self.parameters.trim_links,
            'only_installed': self.parameters.only_installed,
            }

    def transition_check_pydot(self):
        if not HAS_PYDOT:
            self.raise_user_error('no_pydot')
        return 'parameters'

    def transition_graph_(self):
        return 'end'


class ModuleGraph(Report):
    __name__ = 'ir.module.graph'

    @classmethod
    def execute(cls, ids, data):
        pool = Pool()
        ActionReport = pool.get('ir.action.report')

        action_report_ids = ActionReport.search([
            ('report_name', '=', cls.__name__)
            ])
        if not action_report_ids:
            raise Exception('Error', 'Report (%s) not find!' % cls.__name__)
        action_report = ActionReport(action_report_ids[0])

        graph = cls.create_graph()
        cls.fill_graph(graph, data)
        the_graph = graph.create(prog='dot', format='png')
        return ('png', fields.Binary.cast(the_graph), False,
            action_report.name)

    @classmethod
    def create_graph(cls):
        graph = pydot.Dot(fontsize="8")
        graph.set('center', '1')
        graph.set('ratio', 'auto')
        graph.set('rankdir', 'BT')
        graph.set('ranksep', '4')
        return graph

    @classmethod
    def create_node(cls, module):
        # Could be overriden for instance to change color depending on the
        # module state
        return pydot.Node(module.name)

    @classmethod
    def fill_graph(cls, graph, data):
        '''
        We want the minimal dependencies on each module to avoid overly
        complicated graphs.
        '''
        Module = Pool().get('ir.module')

        nodes = {}
        modules = {}
        node_instances = {}
        module_domain = ([('state', '=', 'activated')]
            if data['only_installed'] else [])
        for module in Module.search(module_domain):
            if module.name in ('ugip', 'ugip_migration', 'prevere',
                    'spb_family', 'santiane', 'test_module', 'roederer',
                    'roederer_migration', 'roederer_interface'):
                continue
            node_instances[module.name] = cls.create_node(module)
            graph.add_node(node_instances[module.name])
            nodes[module.name] = [x.name for x in module.parents]
            modules[module.name] = module

        if not data['trim_links']:
            for k, v in nodes.iteritems():
                for dep in v:
                    graph.add_edge(pydot.Edge(k, dep, arrowhead="normal"))
            return

        cache_res = {}

        def get_parents(module, res=None):
            if module.name in cache_res:
                return cache_res[module.name]
            add2cache = False
            if not res:
                res = set([])
                add2cache = True
            for elem in module.parents:
                res.add(elem.name)
                get_parents(elem, res)
            if add2cache:
                cache_res[module.name] = list(res)
                return cache_res[module.name]

        dependencies = {}
        for k, v in nodes.iteritems():
            final, found = dependencies.get(k, [[], []])
            for dep in v:
                if dep in found:
                    continue
                final.append(dep)
                for new_parent in get_parents(modules[dep]):
                    if new_parent in final:
                        final.pop(final.index(new_parent))
                        continue
                    if new_parent in found:
                        continue
                    found.append(new_parent)
                found.append(dep)
            dependencies[k] = [list(set(final)), list(set(found))]

        numbers = {name: [0, node]
            for name, node in node_instances.iteritems()}
        for k, v in dependencies.iteritems():
            for dep in v[0]:
                graph.add_edge(pydot.Edge(k, dep, arrowhead="normal"))
                numbers[dep][0] += 4
            for dep in v[1]:
                numbers[dep][0] += 2

        for _, (number, node) in numbers.iteritems():
            node.set('style', 'filled')
            color = 256 - min(int(number * 256.0 / len(numbers)), 256)
            node.set('fillcolor', '#FF%.2X%.2X' % (color, color))
