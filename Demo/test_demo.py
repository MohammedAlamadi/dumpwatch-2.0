"""Exercise the Streamlit application with real model inference and an isolated database."""
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import sha256
import json
import os
import unittest

from streamlit.testing.v1 import AppTest
from workflow import ROOT, MODELS, Registry, model_path


def element(items, label):
    return next(item for item in items if item.label == label)


class DemoTest(unittest.TestCase):
    def test_live_workflow(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "incidents.sqlite3"
            os.environ["DUMPWATCH_DB"] = str(db)
            self.addCleanup(os.environ.pop, "DUMPWATCH_DB", None)
            app = AppTest.from_file(ROOT / "app.py", default_timeout=120).run()
            self.assertFalse(list(app.exception))

            def button(label):
                element(app.button,label).click().run()
                self.assertFalse(list(app.exception))

            def radio(label,value):
                element(app.radio,label).set_value(value).run()
                self.assertFalse(list(app.exception))

            def footprint(value=True):
                element(app.checkbox,"I confirm that the crops show the same visible ground footprint").set_value(value).run()

            radio("Video source","No-dumping example")
            button("Run live detection")
            self.assertIn("Dumping not detected",[x.value for x in app.info])
            self.assertFalse(Registry(db).list())
            radio("Video source","Upload video")
            examples=json.loads((ROOT/"examples/manifest.json").read_text(encoding="utf8"))
            video=examples["detected"]["file"]
            app.file_uploader[0].set_value((video,(ROOT/"examples"/video).read_bytes(),"video/mp4")).run()
            button("Run live detection")
            result=app.session_state["detection"][1]
            self.assertTrue(result["detected"])
            self.assertGreater(result["score"],0)
            self.assertAlmostEqual(result["onset_seconds"],7.75)
            button("Register incident")
            self.assertFalse(Registry(db).list())
            element(app.text_input,"Camera or location ID (optional)").set_value("DEMO-CAM")
            element(app.checkbox,"I confirm this detection for registration").set_value(True)
            button("Register incident")
            records=Registry(db).list()
            self.assertEqual(len(records),1)
            incident=records[0]["id"]
            self.assertEqual(records[0]["status"],"Awaiting cleanup")

            radio("Inspection source","Upload photographs")
            sample=examples["remains"]
            for label,key in (("Waste-present reference (W)","W"),("Later inspection","inspection")):
                file=sample[key]
                element(app.file_uploader,label).set_value((file,(ROOT/"examples"/file).read_bytes(),"image/jpeg"))
                prefix="w" if key=="W" else "inspection"
                for side,value in zip(("Left","Top","Right","Bottom"),sample[key+"_roi"]):
                    app.number_input(key=f"{prefix}_{side}").set_value(value)
            app.run()
            footprint()
            button("Submit inspection")
            self.assertEqual(Registry(db).list()[0]["status"],"Waste remains")
            detail=json.loads(Registry(db).history(incident)[-1]["detail"])
            self.assertAlmostEqual(detail["score"],-0.7460899329975716)

            radio("Inspection source","Needs review example")
            footprint()
            button("Submit inspection")
            self.assertEqual(Registry(db).list()[0]["status"],"Pending human review")
            self.assertIn("Requires human review",[x.value for x in app.warning])
            detail=json.loads(Registry(db).history(incident)[-1]["detail"])
            self.assertAlmostEqual(detail["score"],0.18272982351238466)
            button("Approve final closure")
            self.assertEqual(Registry(db).list()[0]["status"],"Pending human review")

            radio("Inspection source","Closure review example")
            footprint()
            button("Submit inspection")
            self.assertEqual(Registry(db).list()[0]["status"],"Pending human review")
            element(app.checkbox,"I independently verified that the original waste footprint is clear").set_value(True)
            element(app.text_area,"Review note").set_value("Simulated example: reviewer confirmed the reference footprint is clear.")
            button("Approve final closure")
            fresh=AppTest.from_file(ROOT/"app.py", default_timeout=120).run()
            self.assertFalse(list(fresh.exception))
            self.assertTrue(any(incident in x.value for x in fresh.markdown))
            self.assertEqual(Registry(db).list()[0]["status"],"Verified clean")

    def test_freeze_and_guards(self):
        for name, expected in MODELS.items():
            self.assertEqual(sha256(model_path(name).read_bytes()).hexdigest(),expected)
        with TemporaryDirectory() as tmp:
            registry=Registry(Path(tmp)/"test.sqlite")
            with self.assertRaises(ValueError):
                registry.register({"detected":False},"",True)
            i=registry.register({"detected":True,"simulated":False},"",True)
            with self.assertRaises(ValueError):
                registry.inspection(i,{"outcome":"abstain"},{"sample":True})
            self.assertEqual(registry.list()[0]["status"],"Awaiting cleanup")


if __name__ == "__main__":
    unittest.main(verbosity=2)
