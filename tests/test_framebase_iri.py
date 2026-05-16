from fst2framegraph.framebase.iri import fe_iri, frame_iri


def test_frame_iri_contains_frame_name():
    assert frame_iri("Capability").endswith("/Capability")


def test_fe_iri_uses_has_role():
    assert "has_entity" in fe_iri("Capability", "Entity")
