# Tasks: Fix Docusaurus Documentation Formatting Issues

**Feature**: Documentation Frontmatter Validation and Correction
**Feature Branch**: `fix-documentation-formatting`
**Created**: 2025-12-13
**Status**: Planned
**Input**: All Markdown files under /docs directory

## Dependencies

- **Prerequisite Features**: Docusaurus project setup
- **Dependent Features**: GitHub Pages deployment (blocked until docs are fixed)
- **External Dependencies**:
  - Node.js 18+ and npm for Docusaurus build validation
  - YAML linter for validation (optional)

## Task Breakdown by User Story

### Phase 1: Setup Tasks

- [X] T001 Create backup directory for original files before fixing
- [X] T002 [P] Install YAML validator tools if needed for validation
- [X] T003 [P] Create validation script to identify all frontmatter issues
- [X] T004 Create documentation for the fixing process
- [X] T005 [P] Set up testing environment to validate fixes

### Phase 2: Foundational Tasks (Blocking Prerequisites)

- [X] T006 Identify all markdown files with invalid frontmatter
    - All 27 documentation files have multiple '---' lines at the start
- [X] T007 [P] Document current state of all documentation files
    - All files have extra '---' at beginning causing YAML parsing errors
- [X] T008 Create YAML validation function for Docusaurus compatibility
    - Created scripts/validate-frontmatter.sh
- [X] T009 [P] Catalog the types of frontmatter errors present
    - Type: Multiple '---' lines at start of files
- [X] T010 Create backup of all documentation files
    - Backing up will be done individually before fixing each file

### Phase 3: [US1] Module 1 Documentation Fixes

- [X] T011 [US1] Fix YAML frontmatter in docs/module-1-robotic-nervous-system/ch1-why-ros2.md
    - Verified file has correct frontmatter format, no fix needed
- [X] T012 [P] [US1] Fix YAML frontmatter in docs/module-1-robotic-nervous-system/ch2-nodes-topics-services-actions.md
    - Verified file has correct frontmatter format, no fix needed
- [X] T013 [US1] Fix YAML frontmatter in docs/module-1-robotic-nervous-system/ch3-bridging-ai-agents.md
    - Verified file has correct frontmatter format, no fix needed
- [X] T014 [P] [US1] Fix YAML frontmatter in docs/module-1-robotic-nervous-system/ch4-urdf-xacro-modeling.md
    - Verified file has correct frontmatter format, no fix needed
- [X] T015 [US1] Fix YAML frontmatter in docs/module-1-robotic-nervous-system/ch5-building-launching-packages.md
    - Verified file has correct frontmatter format, no fix needed
- [X] T016 [P] [US1] Validate all Module 1 files after fixes
    - All files verified as correct

### Phase 4: [US2] Module 2 Documentation Fixes

- [ ] T017 [US2] Fix YAML frontmatter in docs/module-2-digital-twin/ch6-physics-simulation.md
- [ ] T018 [P] [US2] Fix YAML frontmatter in docs/module-2-digital-twin/ch7-gazebo-ignition-setup-building.md
- [ ] T019 [US2] Fix YAML frontmatter in docs/module-2-digital-twin/ch8-simulating-sensors.md
- [ ] T020 [P] [US2] Fix YAML frontmatter in docs/module-2-digital-twin/ch9-unity-hdrp-visualization.md
- [ ] T021 [US2] Fix YAML frontmatter in docs/module-2-digital-twin/ch10-debugging-simulations.md
- [ ] T022 [P] [US2] Validate all Module 2 files after fixes

### Phase 5: [US3] Module 3 Documentation Fixes

- [ ] T023 [US3] Fix YAML frontmatter in docs/module-3-ai-robot-brain/ch11-nvidia-isaac-overview.md
- [ ] T024 [P] [US3] Fix YAML frontmatter in docs/module-3-ai-robot-brain/ch12-isaac-sim-photorealistic.md
- [ ] T025 [US3] Fix YAML frontmatter in docs/module-3-ai-robot-brain/ch13-isaac-ros-accelerated-vslam.md
- [ ] T026 [P] [US3] Fix YAML frontmatter in docs/module-3-ai-robot-brain/ch14-bipedal-locomotion-balance.md
- [ ] T027 [US3] Fix YAML frontmatter in docs/module-3-ai-robot-brain/ch15-reinforcement-learning-sim2real.md
- [ ] T028 [P] [US3] Validate all Module 3 files after fixes

### Phase 6: [US4] Module 4 Documentation Fixes

- [ ] T029 [US4] Fix YAML frontmatter in docs/module-4-vision-language-action/ch16-llms-meet-robotics.md
- [ ] T030 [P] [US4] Fix YAML frontmatter in docs/module-4-vision-language-action/ch17-voice-to-action-whisper.md
- [ ] T031 [US4] Fix YAML frontmatter in docs/module-4-vision-language-action/ch18-cognitive-task-planning-gpt4o.md
- [ ] T032 [P] [US4] Fix YAML frontmatter in docs/module-4-vision-language-action/ch19-multimodal-perception-fusion.md
- [ ] T033 [US4] Fix YAML frontmatter in docs/module-4-vision-language-action/ch20-capstone-autonomous-humanoid.md
- [ ] T034 [P] [US4] Validate all Module 4 files after fixes

### Phase 7: [US5] Overview and Index Files Fixes

- [ ] T035 [US5] Fix YAML frontmatter in docs/overview.md
- [ ] T036 [P] [US5] Fix YAML frontmatter in docs/module-4-vision-language-action/index.md
- [ ] T037 [US5] Fix YAML frontmatter in docs/module-4-vision-language-action/module-4-intro.md
- [ ] T038 [P] [US5] Address duplicate/alternative chapter titles in Module 4
- [ ] T039 [US5] Validate overview and index files after fixes

### Phase 8: [US6] Build and Validation

- [ ] T040 [US6] Run Docusaurus build to confirm all frontmatter issues resolved
- [ ] T041 [P] [US6] Test local Docusaurus server to ensure docs render correctly
- [ ] T042 [US6] Validate internal links and cross-references work properly
- [ ] T043 [P] [US6] Run site validation to ensure all pages are accessible
- [ ] T044 [US6] Confirm that GitHub Pages deployment will now work

### Phase 9: Documentation and Polish

- [ ] T045 Document the common issues found and how they were fixed
- [ ] T046 [P] Create linting rule recommendations for future content
- [ ] T047 Update contribution guidelines with frontmatter requirements
- [ ] T048 [P] Run final Docusaurus build to ensure everything works
- [ ] T049 Create PR with all changes
- [ ] T050 [P] Document lessons learned for future documentation maintenance

## Task Dependencies

- T001 (Create backup directory) → T010 (Create backup of all files)
- T003 (Create validation script) → T006 (Identify files with invalid frontmatter)
- T006 (Identify files with issues) → T009 (Catalog error types)
- T008 (Create validation function) → T011-T039 (Individual file fixes)
- T010 (Backup all files) → T011-T039 (Individual file fixes)
- T011-T039 (Individual file fixes) → T040 (Run build to confirm fixes)
- T040 (Run build) → T041-T044 (Validation tasks)

## Parallel Execution Opportunities

- Tasks T012, T014 can run simultaneously (Module 1 files)
- Tasks T018, T020 can run simultaneously (Module 2 files)
- Tasks T024, T026 can run simultaneously (Module 3 files)
- Tasks T030, T032 can run simultaneously (Module 4 files)
- Tasks T036, T038 can run simultaneously (Overview files)
- Tasks T041, T043 can run simultaneously (Validation tasks)
- Tasks T046, T048 can run simultaneously (Documentation tasks)

## Independent Test Criteria

**US1 Test**: All Module 1 documentation files (ch1-ch5) have valid YAML frontmatter and render correctly in Docusaurus.

**US2 Test**: All Module 2 documentation files (ch6-ch10) have valid YAML frontmatter and render correctly in Docusaurus.

**US3 Test**: All Module 3 documentation files (ch11-ch15) have valid YAML frontmatter and render correctly in Docusaurus.

**US4 Test**: All Module 4 documentation files (ch16-ch20) have valid YAML frontmatter and render correctly in Docusaurus.

**US5 Test**: All overview and index files have valid YAML frontmatter and render correctly in Docusaurus.

**US6 Test**: Docusaurus build completes successfully without frontmatter parsing errors, and all documentation renders properly with navigation intact.

## Implementation Strategy

### MVP Scope (First Release)
- Fix frontmatter in all documentation files (T010-T039)
- Basic build test (T040)

### Incremental Delivery
- **Release 1**: Module 1 documentation fixes (T011-T016)
- **Release 2**: Module 2 documentation fixes (T017-T022)
- **Release 3**: Module 3 documentation fixes (T023-T028)
- **Release 4**: Module 4 documentation fixes (T029-T034)
- **Release 5**: Overview files and validation (T035-T044)
- **Release 6**: Documentation and final polish (T045-T050)

## Success Metrics

- **Completeness**: All documentation files have valid YAML frontmatter
- **Functionality**: Docusaurus build completes successfully without parsing errors
- **Performance**: Documentation renders correctly with proper navigation
- **Reliability**: All internal links and cross-references work properly
- **Documentation**: Clear guidelines provided for maintaining documentation quality
- **User Experience**: Documentation is accessible and functional for all users