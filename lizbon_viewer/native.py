"""Insert our presentation layer without changing the pinned native checkout."""
from .doors import DoorLayer


class _IndexView:
    def __init__(self,index,paint):self._index=index;self._paint=paint
    def __getattr__(self,name):return getattr(self._index,name)
    def a2_state(self):
        original=self._index.a2_state()
        # Native MapView asks for A2 state immediately after blitting map art,
        # before trails/bodies/players. Paint here to preserve scene depth.
        self._paint()
        if not original:return original
        return {**original,'doors':{},'local_doors':{}}  # Hide native debug bars only.


class _AppView:
    def __init__(self,app,index):self._app=app;self.index=index
    def __getattr__(self,name):return getattr(self._app,name)


def install():
    from spectator.v2 import renderer
    from spectator.v2.map_view import MapView
    if getattr(renderer.MapView,'lizbon_door_animation',False):return

    class AnimatedMapView(MapView):
        lizbon_door_animation=True
        def __init__(self,world):super().__init__(world);self.lizbon_doors=DoorLayer()
        def draw(self,surface,rect,app,ui):
            if app.index is None:return super().draw(surface,rect,app,ui)
            index=_IndexView(app.index,lambda:self.lizbon_doors.draw(surface,self,app))
            return super().draw(surface,rect,_AppView(app,index),ui)

    renderer.MapView=AnimatedMapView
