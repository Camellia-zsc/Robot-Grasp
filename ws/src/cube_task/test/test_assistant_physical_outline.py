"""A rotated bounding box is not a round body's physical footprint."""
import math

import pytest

from cube_task.assistant.catalog import make_catalog_object
from cube_task.assistant.simulation import SimulationSession


@pytest.mark.parametrize('label,radius',[('pear',.0425),('apple',.031)])
@pytest.mark.parametrize('yaw',[0.,math.pi/4])
def test_region_audit_uses_curved_body_and_still_rejects_real_overhang(tmp_path,label,radius,yaw):
    with SimulationSession(realtime=False,log_dir=tmp_path) as session:
        edge=min(p[1] for p in session._placement['pick_polygon'])
        obj=make_catalog_object(label,'audit-object',[.30,edge+radius+.006],yaw)
        destination={'type':'region','name':'pick_area'}
        assert session._verify_destination(obj,destination)
        moved=obj.T_base_obj
        moved[1,3]-=.012
        obj.T_base_obj=moved
        assert not session._verify_destination(obj,destination)
