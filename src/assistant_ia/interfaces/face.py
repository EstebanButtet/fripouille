"""Rendu Canvas local du contour audité de prototype/vector-eye.

Les deux Bézier viennent de main.c à 81fdc4c ; déformations fonctionnelles
minimales ajoutées sur main, sans prétendre reproduire le prototype Edge absent.
"""
from assistant_ia.expressions import Expression, ExpressiveIntent

_CURVES = (
    ((18, 50), (43, 12), (120, 5), (152, 48)),
    ((152, 48), (130, 88), (47, 95), (18, 50)),
)


def eye_contour(expression: Expression, *, right: bool = False) -> tuple[float, ...]:
    if not isinstance(expression, Expression):
        raise TypeError("Unknown face expression.")
    scale_y = {Expression.NEUTRAL: 1, Expression.FOCUSED: .5,
               Expression.CURIOUS: 1.15 if right else .8,
               Expression.CONCERNED: .7}[expression]
    points = []
    for curve in _CURVES:
        for step in range(17):
            t = step / 16
            weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
            x, y = (sum(w*p[axis] for w, p in zip(weights, curve)) for axis in (0, 1))
            points.extend(((170-x if right else x)*.43 + (90 if right else 8),
                           (y-50)*scale_y*.43 + 70))
    return tuple(points)


class CanvasFacePresenter:
    def __init__(self, canvas):
        self.canvas = canvas

    def present_expression(self, intent: ExpressiveIntent) -> None:
        if not isinstance(intent, ExpressiveIntent):
            raise TypeError("A validated expressive intent is required.")
        self.canvas.delete("expression")
        for right in (False, True):
            self.canvas.create_line(*eye_contour(intent.expression, right=right),
                                    fill="#53F3FF", width=3, tags="expression")
        self.canvas.create_line(65, 118, 105, 118, fill="#53F3FF", width=3, tags="expression")
