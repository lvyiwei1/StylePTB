from collections import namedtuple
from itertools import izip

from gtd.text import word_to_forms


class EditPath(namedtuple('EditPath', ['s', 't', 'actions', 'costs'])):
    def __repr__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        pr1 = []
        pr2 = []
        pr3 = []
        s = list(reversed(self.s))
        t = list(reversed(self.t))
        for dec, cost in izip(self.actions, self.costs):
            if dec == 'i':
                p1 = u''
                p2 = t.pop()
                p3 = p2
            elif dec == 'd':
                p1 = s.pop()
                p2 = u''
                p3 = u'D'
            else:
                p1 = s.pop()
                p2 = t.pop()
                p3 = u'C'  # copy

            if (p1 and p2) and p1 != p2:
                p2 = u'[{}]'.format(p2)
                p3 = u'T'  # transform

            # note any non-zero costs
            if cost != 0:
                p3 = u'{}({})'.format(p3, int(cost))

            d = max(len(p1), len(p2), len(p3))
            fmt = lambda x: (u'{:^%d}' % d).format(x)

            pr1.append(fmt(p1))
            pr2.append(fmt(p2))
            pr3.append(fmt(p3))

        j = lambda elems: u' '.join(elems)
        return u'\n'.join([j(pr1), j(pr2), j(pr3)])


def edit_dist(s, t, transform_cost=None, delete_cost=None, insert_cost=None):
    """Compute edit distance.

    Args:
        s (list): source sequence
        t (list): target sequence
        transform_cost (Callable[[object, object], float]): function f(x, y) = cost of transforming x into y
        delete_cost (Callable[object, float]): function f(x) = cost of deleting x
        insert_cost (Callable[object, float]): function f(x) = cost of inserting x

    Returns:
        min_cost (int)
        path (list[str]): sequence consisting of:
            d = delete
            i = insert
            t = transform
    """
    if transform_cost is None:
        transform_cost = lambda x, y: 0.0 if x == y else 1.0
    if delete_cost is None:
        delete_cost = lambda x: 1.0
    if insert_cost is None:
        insert_cost = lambda x: 1.0

    m, n = len(s), len(t)
    dp = [[None] * (n + 1) for _ in xrange(m + 1)]
    inf = float('inf')

    # each DP cell holds the minimum value, plus the optimal decision
    for i in xrange(m, -1, -1):
        for j in xrange(n, -1, -1):
            s_gone = (i == m)  # s is empty
            t_gone = (j == n)  # t is empty

            if s_gone and t_gone:
                # base case
                result = (0, None)
            else:
                if s_gone:
                    delete = (inf, 'd')  # cannot delete when s is all gone
                else:
                    delete = (dp[i + 1][j][0] + delete_cost(s[i]), 'd')

                if t_gone:
                    insert = (inf, 'i')  # cannot insert when t is all done
                else:
                    insert = (dp[i][j + 1][0] + insert_cost(t[j]), 'i')

                if not s_gone and not t_gone:
                    next_cost = dp[i + 1][j + 1][0]
                    transform = (next_cost + transform_cost(s[i], t[j]), 't')
                else:
                    # can't transform
                    transform = (inf, None)

                # TODO: break ties better
                result = min(delete, transform, insert)

            dp[i][j] = result

    # recover optimal trajectory
    actions = []
    costs = []
    i, j = 0, 0
    while True:
        cost_left, dec = dp[i][j]
        if dec is None:
            break

        actions.append(dec)

        if dec == 'd':
            i += 1
        elif dec == 'i':
            j += 1
        elif dec == 't':
            # transform
            i += 1
            j += 1
        else:
            raise RuntimeError('Invalid decision: {}'.format(dec))

        # cost of this decision is the reduction in cost left after the decision
        cost = cost_left - dp[i][j][0]
        costs.append(cost)

    return dp[0][0][0], EditPath(s, t, actions, costs)


STOPWORDS = ['all', 'just', 'being', 'over', 'both', 'month', 'through',
             'during', 'its', 'before', 'group', 'with', 'had', 'than', 'to',
             'only', 'minute', 'under', 'ours', 'has', 'do', 'them', 'his',
             'around', 'very', 'they', 'not', 'yourselves', 'now', 'him', 'nor',
             'like', 'did', 'should', 'this', 'she', 'each', 'further', 'where',
             'few', 'because', 'century', 'people', 'doing', 'theirs', 'some',
             'are', 'year', 'our', 'beyond', 'ourselves', 'out', 'what', 'for',
             'since', 'while', 'behind', 'does', 'above', 'between', 'across',
             't', 'be', 'we', 'after', 'here', 'hers', 'along', 'by', 'on',
             'about', 'of', 'against', 's', 'place', 'or', 'among', 'own',
             'into', 'within', 'yourself', 'down', 'throughout', 'your', 'city',
             'from', 'her', 'whom', 'there', 'due', 'been', 'their', 'much',
             'too', 'themselves', 'was', 'until', 'more', 'himself', 'that',
             'but', 'don', 'herself', 'below', 'those', 'he', 'me', 'myself',
             'hour', 'these', 'type', 'up', 'will', 'near', 'can', 'were',
             'country', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as',
             'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when',
             'same', 'how', 'other', 'which', 'you', 'many', 'day', 'towards',
             'who', 'upon', 'most', 'date', 'such', 'why', 'a', 'off', 'i',
             'having', 'person', 'without', 'so', 'the', 'yours', 'once', ',', ';', '.']

# ensure they are all lower case
for word in STOPWORDS:
    assert word == word.lower()


class CustomEditDistance(object):
    """Customized edit distance, see details below.
    
    Deletions are free.
    
    Transformations between words of the same lemma are free.
    All other transformations cost <penalty>.
    
    Insertion costs 1 if the word is a stopword or appears in a provided insertable set.
    All other insertions cost <penalty>.
    """
    def __init__(self, penalty=10.):
        self.penalty = penalty
        self.delete_cost = lambda x: 0.0
        self.transform_cost = lambda x, y: 0.0 if len(word_to_forms(x) & word_to_forms(y)) != 0 else penalty
        self.stopwords = set(STOPWORDS)

    def __call__(self, s, t, insertable):
        """Compute edit distance.
        Args:
            s (list[unicode]): source sequence
            t (list[unicode]): target sequence
            insertable (set[unicode]): words that can be inserted at no cost

        Returns:
            dist (float)
            path (EditPath)
        """
        penalty = self.penalty
        stopwords = self.stopwords

        def insert_cost(x):
            if x in stopwords or x in insertable:
                return 1.
            return penalty

        return edit_dist(s, t, transform_cost=self.transform_cost,
                         delete_cost=self.delete_cost, insert_cost=insert_cost)